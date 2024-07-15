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

dirpath = "/Users/osoucy/Documents/okube/databricks-exam-questions/exam-03"

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

    print(f"Reading {i:04d}/{n:04d} {filename}... ", end="")

    with Image.open(filepath) as img:
        image = preprocess_image(img)

    text = pytesseract.image_to_string(image)

    print(f" Parsing... ", end="")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system",
             "content": "You are a text parser and you need to process text extracted from a image and return a structured json string storing one question and the associated 4 multiple choices."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )

    d = json.loads(completion.choices[0].message.content)

    with open(filepath.replace(".png", ".json"), "w") as fp:
        json.dump(d, fp, indent=4)

    print(f"done.")
    time.sleep(1.0)
