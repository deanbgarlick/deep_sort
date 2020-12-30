python deep_sort/tools/generate_detections.py \
    --model=mars-small128.pb \
    --mot_dir=./input_data \
    --output_dir=./detections

python deep_sort/deep_sort_app.py \
    --sequence_dir=./input_data/sequence-1 \
    --detection_file=./detections/sequence-1.npy \
    --output_file=./output/sequence-1.txt \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=False