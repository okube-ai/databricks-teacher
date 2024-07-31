from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
import pytesseract
import time
import os
import json
import openai
import yaml


# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #

exam_type = "engineer-professional"
exam_number = "05"
dirpath = f"/Users/osoucy/Documents/okube/databricks-exam-questions/{exam_type}-exam-{exam_number}"

client = openai.OpenAI(
    organization="org-SaovQqDsnEobPkVB0C8Adt1A",
)


# --------------------------------------------------------------------------- #
# Functions                                                                   #
# --------------------------------------------------------------------------- #


def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert("L")
    # Resize image
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2)
    # Apply binary thresholding
    image = image.point(lambda x: 0 if x < 140 else 255, "1")
    # Apply median filter
    image = image.filter(ImageFilter.MedianFilter())
    # Invert image (optional, depends on the quality of the original image)
    image = ImageOps.invert(image)
    return image


# --------------------------------------------------------------------------- #
# Questions Overwrite                                                         #
# --------------------------------------------------------------------------- #

with open("questions_overwrite.yaml") as fp:
    overwrites = yaml.safe_load(fp)


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #

filenames = list(os.listdir(dirpath))
filenames = [f for f in filenames if f.endswith(".png")]

n = len(filenames)
for i, filename in enumerate(filenames):

    filepath = os.path.join(dirpath, filename)
    output_filepath = filepath.replace(".png", ".json")

    if os.path.exists(output_filepath):
        continue

    print(f"Reading {i:04d}/{n:04d} {filename}... ", end="")

    with Image.open(filepath) as img:
        image = preprocess_image(img)

    text = pytesseract.image_to_string(image)

    print(f" Parsing... ", end="")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": """
             You are a text parser and you need to process text extracted from a image and return json string with
            this format:
            {"question": "...", "choices": {"A": "...", "B": "...", "C":"...", "D":"...", "E":"..."}}
            """,
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    try:
        d = json.loads(completion.choices[0].message.content)
    except json.decoder.JSONDecodeError:
        continue

    d["question_number"] = filename.replace(
        "q",
        "",
    ).replace(".png", "")
    d["question_number"] = int(d["question_number"])
    d["exam_type"] = exam_type
    d["exam_number"] = int(exam_number)
    d["section_index"] = None

    # Overwrite
    for o in overwrites:
        if (
            d["exam_type"] == o["exam_type"]
            and d["exam_number"] == o["exam_number"]
            and d["question_number"] == o["question_number"]
        ):
            for k in ["choices", "section_index"]:
                if k in o:
                    d[k] = o[k]

    try:
        d["question"]
        d["choices"]["A"]
        d["choices"]["B"]
        d["choices"]["C"]
        d["choices"]["D"]
        d["choices"]["E"]
    except KeyError as e:
        print(exam_type, exam_number, d["question_number"])
        print(text)
        print(d)
        raise e

    with open(filepath.replace(".png", ".json"), "w") as fp:
        json.dump(d, fp)

    print(f"done.")
    time.sleep(1.0)
