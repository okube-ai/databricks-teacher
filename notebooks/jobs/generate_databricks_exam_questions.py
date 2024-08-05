# COMMAND ----------
# MAGIC %pip install 'laktory==0.4.10' langchain langchain_community 'pdfminer.six'

# COMMAND ----------
import time
import json
import uuid
from datetime import datetime
from collections import defaultdict
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from laktory import models

# --------------------------------------------------------------------------- #
# Read Data                                                                   #
# --------------------------------------------------------------------------- #

df = spark.read.table("dev.databricks.slv_exam_sections")
pdf = df.toPandas()
run_id = str(uuid.uuid4())

# ----------------------------------------------------------------------- #
# Set Prompt and Chain                                                    #
# ----------------------------------------------------------------------- #

prompt_template = """
You are Databricks Certification Exam questions generator. 
You need to create completely new questions given the exam content and some sample questions. You can't re-use the sample questions. You must also avoid the provided excluded questions.

Exam Content: {content}

Sample Questions: {questions}

Excluded Questions: {excluded_questions}

Your answer should be only a valid json list, with {n} items following this structure:
    {{
        "question": question,
        "A": choice A,
        "B": choice B,
        "C": choice C,
        "D": choice D,
        "E": choice E,
        "answer": correct answer,
        "explanation": explanation,
    }}
Ensure each question is completely different from the others.
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["content", "questions", "excluded_questions", "n"])

# Set LLM
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.0,
    top_k=0.95,
)

# Set Chain
chain = prompt | llm

# ----------------------------------------------------------------------- #
# Build Data                                                              #
# ----------------------------------------------------------------------- #

data = defaultdict(lambda: [])

irow = -1
for _, row in pdf.iterrows():
    irow += 1

    section_index = row["index"]
    section_id = row["id"]
    section_name = row["name"]
    exam_type = row["exam_type"]
    section_questions = []

    if row["exam_type"] not in ["analyst-associate", "engineer-associate"]:
        continue

    print(f"Processing {exam_type} - {section_name}")

    for i in range(5):

        t0 = time.time()
        print(f"   Invoking LLM for chunk {i}... ", end="")
        answer = chain.invoke({
            "questions": row["questions_and_answers"],
            "content": row["content"],
            "excluded_questions": section_questions,
            "n": 10,
        })
        print(f"completed. [{time.time() - t0:5.2f}]")
        created_at = datetime.utcnow()

        try:
            content = answer.content
            content = content.split("[")[1:]
            content = "[" + "[".join(content)
            questions = json.loads(content)
        except Exception as e:
            print(answer.content)
            raise e

        for q in questions:
            section_questions += [q["question"]]
            event = models.DataEvent(
                name="databricks_exam_question",
                producer=models.DataProducer(name="okube"),
                data={
                    "uuid": str(uuid.uuid4()),
                    "run_id": run_id,
                    "created_at": created_at,
                    "exam_type": exam_type,
                    "section_index": section_index,
                    "section_name": section_name,
                    "section_id": section_id,
                    "question": q["question"],
                    "A": q["A"],
                    "B": q["B"],
                    "C": q["C"],
                    "D": q["D"],
                    "E": q["E"],
                    "answer": q["answer"],
                    "explanation": q["explanation"]
                },
            )
            event.to_databricks(suffix=f"{section_id}-{len(section_questions)-1:03d}")
