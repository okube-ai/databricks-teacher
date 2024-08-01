import yaml
import json
import os

# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #

exam_ids = [
    "analyst-associate-5",
    "analyst-associate-6",
    "engineer-associate-1",
    "engineer-associate-2",
    "engineer-professional-1",
    "engineer-professional-2",
    "engineer-professional-3",
    "engineer-professional-4",
    "engineer-professional-5",
]

# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #

for exam_id in exam_ids:
    dirpath = f"./data/"
    filename = exam_id + ".yaml"
    filepath = os.path.join(dirpath, filename)

    print(f"Processing {filename}")
    with open(filepath, "r") as fp:
        data = yaml.safe_load(fp)

    with open(filepath.replace(".yaml", ".json"), "w") as fp:
        json.dump(data, fp)
