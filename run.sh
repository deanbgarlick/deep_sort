python deep_sort/tools/generate_detections.py \
    --model=mars-small128.pb \
    --mot_dir=./data \
    --output_dir=./detections

python deep_sort/deep_sort_app.py \
    --sequence_dir=./data/sequence-1 \
    --detection_file=./detections/sequence-1.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=False