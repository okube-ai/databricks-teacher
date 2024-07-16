from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
import pytesseract
import time
import os
import json
import openai


# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #

exam_type = "analyst-associate"
exam_number = "04"
dirpath = f"/Users/osoucy/Documents/okube/databricks-exam-questions/{exam_type}-exam-{exam_number}"

client = openai.OpenAI(organization="org-SaovQqDsnEobPkVB0C8Adt1A",)


# --------------------------------------------------------------------------- #
# Functions                                                                   #
# --------------------------------------------------------------------------- #

def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Resize image
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2)
    # Apply binary thresholding
    image = image.point(lambda x: 0 if x < 140 else 255, '1')
    # Apply median filter
    image = image.filter(ImageFilter.MedianFilter())
    # Invert image (optional, depends on the quality of the original image)
    image = ImageOps.invert(image)
    return image


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
            {"role": "system",
             "content": """
             You are a text parser and you need to process text extracted from a image and return json string with
            this format:
            {"question": "...", "choices": {"A": "...", "B": "...", "C":"...", "D":"..."}}
            """
             },
            {"role": "user", "content": text}
        ],
        temperature=0
    )

    d = json.loads(completion.choices[0].message.content)
    d["question_number"] = filename.replace("q", "",). replace(".png", "")
    d["exam_type"] = exam_type
    d["exam_number"] = exam_number

    if exam_type == "analyst-associate":

        if exam_number == "02" and d["question_number"] == "053":
            d["choices"]["D"] = "The amount of training data available"

        if (exam_number == "03" and d["question_number"] == "023") or (exam_number == "04" and d["question_number"] == "161"):
            d["choices"]["D"] = d["choices"]["C"]
            d["choices"]["C"] = d["choices"]["B"]
            d["choices"]["B"] = d["choices"]["A"]
            d["choices"]["A"] = "Use Scikit-learn for rapid prototyping and evaluate using R-squared."

        if (exam_number == "03" and d["question_number"] == "060") or (exam_number == "04" and d["question_number"] == "139"):
            d["choices"] = {
                "A": "Databricks Visualizations with static widgets",
                "B": "Matplotlib embedded in HTML pages",
                "C": "Plotly Dash for creating interactive web applications",
                "D": "Seaborn plots within a Jupyter notebook",
            }

    # if exam_type == "analyst-associate" and exam_number == "03" and d["question_number"] == "060":

    try:
        d["question"]
        d["choices"]["A"]
        d["choices"]["B"]
        d["choices"]["C"]
        d["choices"]["D"]
    except KeyError as e:
        print(exam_type, exam_number, d["question_number"])
        print(text)
        print(d)
        raise e

    with open(filepath.replace(".png", ".json"), "w") as fp:
        json.dump(d, fp)

    print(f"done.")
    time.sleep(1.0)
