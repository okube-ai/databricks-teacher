# Databricks notebook source
# MAGIC #%pip install linkedin-api
# MAGIC %pip install mlflow langchain langchain_community
# MAGIC %restart_python

# COMMAND ----------
import random
import pyspark.sql.functions as F
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

# Data
catalog_name = "dev"
schema_name = "databricks"
endpoint_name = "databricks-exams"
table_name = f"{catalog_name}.{schema_name}.slv_exam_questions"

# LLM
prompt_template = """
You need to summarize:
- a question with a hard limit of 140 characters
- option A with a hard limit of 30 characters
- option B with a hard limit of 30 characters
- option C with a hard limit of 30 characters
- option D with a hard limit of 30 characters
You can use abbreviations or truncation to be as concise as required. Never go beyond the character limits.


Question: {question}
Option A: {A}
Option B: {B}
Option C: {C}
Option D: {D}

Provide your answer with this format:
Q: {{question summary (140 chars max)}}
A: {{option A summary (30 chars max)}}
B: {{option B summary (30 chars max)}}
C: {{option C summary (30 chars max)}}
D: {{option D summary (30 chars max)}}
"""
prompt = PromptTemplate(template=prompt_template,
                        input_variables=["question", "A", "B", "C", "D"])

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.0,
    top_k=0.95,
)
chain = prompt | llm

# Posted
posteds = [
    "abccc487-9e09-43c6-b0b4-d817ea39c1ec",
    "443cfc6f-1456-49e4-891e-129ff89216db",
    "bb28ab01-adba-4f16-8626-2f9faa717888",
    "3da45980-3d6a-4ab4-8d8c-5b6ae72e41fb",  # does not make sense
    "aead4be0-6024-43c4-aa3b-6d16ec326922",
    "4a7bd3b8-35ed-4297-99c4-ac4cbd94ca92",
]

# Read Data
df = (
    spark.read.table(table_name)
    .sort("section_id", "created_at")
    .toPandas()
)
# display(df)

# COMMAND ----------
# Post Message
_df = df[~df["uuid"].isin(posteds)]
n = len(_df)
iloc = random.randrange(0, n)
# iloc = 40
q = df.iloc[iloc]
exam_type = q["exam_type"]
exam_name = {
    "analyst-associate": "Data Analyst Associate",
    "engineer-associate": "Data Engineer Associate",
    "engineer-professional": "Data Engineer Professional",
    "ml-associate": "Machine Learning Associate",
    "ml-professional": "Machine Learning Professional",
    "genai-associate": "Generative AI Associate",
}[exam_type]
section_index = q["section_index"]
section_name = q["section_name"]
question = q["question"]
answer = q["answer"]
A = q["A"]
B = q["B"]
C = q["C"]
D = q["D"]
E = q["E"]
if answer == "E":
    A = E
    answer = "E"

hash_tags = "#databricks #databrickscertified #dataanalyst #dailyquiz"
if "engineer" in exam_type:
    hash_tags = hash_tags.replace("#dataanalyst", "#dataengineer")

print(f"Selected Question: {iloc:04d} - {exam_type} - {section_index}. {section_name} - {q['uuid']}")

post = f"""Databricks Certifications: Test Your Knowledge! 🚀
↪️ Exam: {exam_name}
↪️ Section: {section_index} - {section_name}

❓{question}

A. {A}
B. {B}
C. {C}
D. {D}

The answer will be posted in the comments below!

🔹 Follow me for daily Databricks certification questions and other great data engineering content! 

{hash_tags}"""
print("-----")
print(post)
print("-----")

# Post Poll
summary = chain.invoke({"question": question, "A": A, "B": B, "C": C, "D": D})
print(summary.content)
print("-----")
print(q)

# COMMAND ----------
# from linkedin_api import Linkedin
# # Authenticate using your LinkedIn credentials
# client = Linkedin(
#     username="olivier.soucy@gmail.com",
#     password=""
# )
#
# # Define the poll options
# poll_options = [
#     {"text": "Option 1"},
#     {"text": "Option 2"},
#     {"text": "Option 3"},
#     {"text": "Option 4"}
# ]
#
# # Create a post with a poll
# post_content = {
#     "author": "urn:li:person:olivier-soucy-2893a21b",
#     "commentary": "What is your favorite option?",
#     "lifecycleState": "PUBLISHED",
#     "visibility": {
#         "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
#     },
#     "content": {
#         "poll": {
#             "question": "What is your favorite option?",
#             "options": poll_options,
#             "endDatetime": datetime.utcnow() + timedelta(days=3)
#         }
#     }
# }
#
# # Post the content to LinkedIn
# response = client.submit_post(post_content)
#
# # Print response
# print(response)
