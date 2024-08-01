import io
import re
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd


@F.pandas_udf(T.StringType())
def binary_to_text(data: pd.Series) -> pd.Series:
    from pdfminer.high_level import extract_text

    def _binary_to_text(data):
        text = extract_text(io.BytesIO(data))
        text = re.sub(r" ?\.", ".", text)
        text = re.sub("", ".", text)

        return text

    return data.apply(_binary_to_text)


#
# @F.pandas_udf(T.ArrayType(T.StringType()))
# def binary_to_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
#
#     def binary_to_text(data):
#         text = extract_text(io.BytesIO(data))
#         text = re.sub(r" ?\.", ".", text)
#         text = re.sub("", ".", text)
#
#         return text
#
#     # Set llama2 as tokenizer
#     set_global_tokenizer(
#         AutoTokenizer.from_pretrained(
#             "hf-internal-testing/llama-tokenizer",
#             cache_dir="/Workspace/tmp/huggingface/",
#         )
#     )
#
#     # Sentence Splitter from llama_index to split on sentences
#     splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
#
#     def extract_and_split(b):
#         text = binary_to_text(b)
#         nodes = splitter.get_nodes_from_documents([Document(text=text)])
#         return [n.text for n in nodes]
#
#     for x in batch_iter:
#         yield x.apply(extract_and_split)
#
#
# @F.pandas_udf(T.ArrayType(T.FloatType()))
# def embed(contents: pd.Series) -> pd.Series:
#     import mlflow.deployments
#
#     deploy_client = mlflow.deployments.get_deploy_client("databricks")
#
#     def get_embeddings(batch):
#         response = deploy_client.predict(
#             endpoint="databricks-bge-large-en", inputs={"input": batch}
#         )
#         return [e["embedding"] for e in response.data]
#
#     # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
#     max_batch_size = 150
#     batches = [
#         contents.iloc[i : i + max_batch_size]
#         for i in range(0, len(contents), max_batch_size)
#     ]
#
#     # Process each batch and collect the results
#     all_embeddings = []
#     for batch in batches:
#         all_embeddings += get_embeddings(batch.tolist())
#
#     return pd.Series(all_embeddings)
#
#
# @F.pandas_udf(T.StringType())
# def row_id(file: pd.Series, content: pd.Series) -> pd.Series:
#     import hashlib
#     def short_hash(text, length=16):
#         text_bytes = text.encode('utf-8')
#         sha256_hash = hashlib.sha256(text_bytes).hexdigest()
#         return sha256_hash[:length]
#
#     return (file + "_" + content).apply(short_hash)


@F.pandas_udf(T.StringType())
def binary_to_text(data: pd.Series) -> pd.Series:
    from pdfminer.high_level import extract_text

    def _binary_to_text(data):
        text = extract_text(io.BytesIO(data))
        text = re.sub(r" ?\.", ".", text)
        text = re.sub("", ".", text)

        return text

    return data.apply(_binary_to_text)


@F.udf(returnType=T.StringType())
def summarize_guide(content: str) -> str:

    from langchain.docstore.document import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chat_models import ChatDatabricks

    # Create Documents
    docs = [Document(page_content=content)]

    # Set Prompt
    prompt_template = """Write a concise summary of the following:


    {text}


    Just provide the summary without additional context:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Set LLM
    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")

    # Set Chain
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=prompt,
        verbose=False,
    )

    summary = chain.invoke(docs)

    return summary["output_text"]


@F.udf(returnType=T.StringType())
def get_section_content(content: str, section: str) -> str:

    from langchain.docstore.document import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chat_models import ChatDatabricks

    # Create Documents
    docs = [Document(page_content=content)]

    # Set Prompt
    prompt_template = f"""Extract the full content of the section named {section}
    in the following:
    {{text}}


    Just provide the section content without additional context:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Set LLM
    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")

    # Set Chain
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=prompt,
        verbose=False,
    )

    summary = chain.invoke(docs)

    return summary["output_text"]

#
# @F.udf(returnType=T.StringType())
# def get_question_section(question: str, content: str) -> str:
#
#     from langchain.docstore.document import Document
#     from langchain.prompts import PromptTemplate
#     from langchain.chat_models import ChatDatabricks
#
#     prompt_template = f"""
#     You will be provide with an exam question and the exam documentation and
#     you need to identify to which section of the exam the question belongs to.
#
#     Exam documentation: {content}
#
#     Exam Question: {{question}}
#
#     Provide only the section using the following format: section_number-section_name
#     """
#
#     prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
#
#     # Set LLM
#     llm = ChatDatabricks(endpoint="databricks-dbrx-instruct")
#
#     # Set Chain
#     chain = prompt | llm
#
#     section = chain.invoke({"question": question})
#
#     return str(section.content)
#
#
# @F.udf(returnType=T.StringType())
# def get_question_section_index(text: str) -> str:
#     text = text.split("-")[0]
#     try:
#         return int(text) - 1
#     except ValueError:
#         return None


def join_questions(self, other) -> str:
    other = other.groupby("section_id").agg(
        F.collect_list("question_and_answers").alias("questions_and_answers"),
    ).withColumnRenamed("section_id", "id")
    return self.join(other, how="left", on="id")

