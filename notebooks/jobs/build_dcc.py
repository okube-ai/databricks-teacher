# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch langchain langchain-community
# MAGIC %restart_python

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatDatabricks
from langchain.embeddings import DatabricksEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser
from mlflow.models import infer_signature
import langchain
import langchain_community
import mlflow

# COMMAND ----------
# Setup
catalog_name = "dev"
schema_name = "databricks"
endpoint_name = "databricks-exams"
index_full_name = f"{catalog_name}.{schema_name}.slv_exam_guides_vs_index"


# COMMAND ----------
# Define Retriever
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")


def get_retriever(persist_dir: str = None):
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=endpoint_name,
        index_name=index_full_name,
    )

    vector_store = DatabricksVectorSearch(
        vs_index,
        text_column="content",
        embedding=embedding_model,
    )

    return vector_store.as_retriever(search_kwargs={"k": 6})


# Test Retriever
vectorstore = get_retriever()
similar_documents = vectorstore.invoke(
    "What are the main topics of Data Engineering Professional certification?"
)
print(f"Relevant documents: {similar_documents}")


# COMMAND ----------
# Define LLM Chain
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat")

TEMPLATE = """
You are Databricks certification assistant used by data specialists who
intend in getting one of the following certifications:
- Data Analyst Associate
- Data Engineer Associate
- Data Engineer Professsioal
- Generative AI Associate
- Machine Learning Associate
- Machine Learning Professional

You should be able to:
- Provide guidance on exam procedure
- Provide and explain the content of each exam

Using this piece of context:
<context>
{context}
</context>

Question: {question}
"""

prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt},
)

# Test Chain
question = {
    "query": "What are the main topic of the Data Analyst Associate certifications?"
}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------
# Register MLFlow Model
mlflow.set_registry_uri("databricks-uc")
model_name = "databricks_certification_coach"
model_full_name = f"dev.databricks.{model_name}"

question = {
    "query": "What are the main topic of the Data Analyst Associate certifications?"
}
answer = chain.invoke(question)

with mlflow.start_run(run_name=model_name):
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,
        artifact_path="chain",
        registered_model_name=model_full_name,
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"langchain=={langchain.__version__}",
            f"langchain-community=={langchain_community.__version__}",
            f"databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature,
    )
