
# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from deep_sort.application_util import preprocessing
from deep_sort.application_util import visualization
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

import base64
import datetime
import pickle

import numpy as np
import cv2
import tensorflow as tf

from kafka import KafkaConsumer, KafkaProducer
from PIL import Image

from deep_sort.deep_sort.iou_matching import iou
from deep_sort.application_util.visualization import ImageViewerStream



def from_base64(buf):
    buf_decode = base64.b64decode(buf)
    buf_arr = np.fromstring(buf_decode, dtype=np.uint8)
    return cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)


topic_in = "object-detections"
topic_out = "object-tracking-video"


def main():

    model_file = 'mars-small128.pb'
    score_threshold = 0.65
    nn_budget = None
    max_cosine_distance = 0.2
    min_confidence = 0.8
    nms_max_overlap = 1.0
    min_detection_height = 0
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    DEFAULT_UPDATE_MS = 20

    consumer = KafkaConsumer(
        topic_in,
        bootstrap_servers=['kafka1:19091']
    )

    # Start up producer
    producer = KafkaProducer(bootstrap_servers='kafka1:19091')

    results = []

    encoder = create_box_encoder(model_file, batch_size=32)

    image_id = -1

    for msg in consumer:

        buf = base64.b64decode(msg.value)
        payload = pickle.loads(buf)
        image_id, img, predictions = payload['image_id'], payload['img'], payload['frame_results']

        img = from_base64(img)
        image = Image.fromarray(img)

        predictions = predictions[0]
        predicions = predictions[predictions[:, 6] > score_threshold, :]

        detections = generate_detections(encoder, image, predictions)

        results += run(
            image_id, image, detections,
            metric, tracker, min_confidence,
            nms_max_overlap, min_detection_height,
            max_cosine_distance, nn_budget)

        seq_info = {
            "min_frame_idx": 0,
            "max_frame_idx": np.inf,
            "update_ms": DEFAULT_UPDATE_MS,
            "image_filenames": 'video_stream',
            "image_size": img.shape,
            "feature_dim": detections.shape[1] - 10 if detections is not None else 0,
        }

        # # Store results.
        # f = open(output_file, 'w')
        # for row in results:
        #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
        #         row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

        visualizer = visualization.StreamVisualization(seq_info, seq_info["update_ms"])

        img_output = visualizer.run(frame_annotation_callback, image_id, image, detections, results)

        # Convert image to jpg
        _, buffer = cv2.imencode('.jpg', img_output)
        producer.send(topic_out, buffer.tobytes())



def frame_annotation_callback(vis, frame_idx, image, detections, results):
    print("Frame idx", frame_idx)

    vis.set_image(image.copy())

    if detections is not None:
        vis.draw_detections(detections)

    mask = results[:, 0].astype(np.int) == frame_idx
    track_ids = results[mask, 1].astype(np.int)
    boxes = results[mask, 2:6]
    vis.draw_groundtruth(track_ids, boxes)


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, image, detections_in):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """

    detections_out = []

    frame_indices = detections_in[:, 0].astype(np.int)
    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()
    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]
        #bgr_image = cv2.imread(
        #    image_filenames[frame_idx], cv2.IMREAD_COLOR)
        #features = encoder(bgr_image, rows[:, 2:6].copy())
        features = encoder(image, rows[:, 2:6].copy())
        detections_out += [np.r_[(row, feature)] for row, feature
                            in zip(rows, features)]

    return np.asarray(detections_out)


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(frame_idx, image, detections, metric, tracker, min_confidence, nms_max_overlap, min_detection_height,
        max_cosine_distance, nn_budget):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """

    results = []

    # Load image and generate detections.
    detections = create_detections(
        detections, frame_idx, min_detection_height)
    detections = [d for d in detections if d.confidence >= min_confidence]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Update tracker.
    tracker.predict()
    tracker.update(detections)
    # Store results.
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlwh()
        results.append([
            frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    return results


if __name__ == "__main__":
    main()
