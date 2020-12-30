import os
import subprocess


def main():
    subprocess.call(
        [
            "python", "deep_sort/tools/generate_detections.py", \
            "--model=mars-small128.pb", \
            "--mot_dir=./input_data", \
            "--output_dir=./detections"
        ]
    )

    for sequence in os.listdir('input_data'):

        if sequence not in ['.DS_Store', 'placeholder']:

            subprocess.call(
                [
                    "python", "deep_sort/deep_sort_app.py", \
                    "--sequence_dir=./input_data/{sequence}".format(sequence=sequence), \
                    "--detection_file=./detections/{sequence}.npy".format(sequence=sequence), \
                    "--output_file=./output/{sequence}.txt".format(sequence=sequence), \
                    "--min_confidence=0.3", \
                    "--nn_budget=100", \
                    "--display=False"
                ]
            )

    subprocess.call(
        [
            "python", "deep_sort/generate_videos.py", \
            "--mot_dir=./input_data", \
            "--result_dir=./output", \
            "--output_dir=./final"
        ]
    )


if __name__ == "__main__":
    main()