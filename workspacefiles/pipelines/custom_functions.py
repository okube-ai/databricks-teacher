import io
import re
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
from typing import Iterator

from pdfminer.high_level import extract_text
from transformers import AutoTokenizer
from llama_index.core import Document
from llama_index.core.utils import set_global_tokenizer
from llama_index.core.langchain_helpers.text_splitter import SentenceSplitter


@F.pandas_udf(T.ArrayType(T.StringType()))
def binary_to_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:

    def binary_to_text(data):
        text = extract_text(io.BytesIO(data))
        text = re.sub(r" ?\.", ".", text)
        text = re.sub("", ".", text)

        return text

    # Set llama2 as tokenizer
    set_global_tokenizer(
        AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            cache_dir="/Workspace/tmp/huggingface/",
        )
    )

    # Sentence Splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)

    def extract_and_split(b):
        text = binary_to_text(b)
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)


@F.pandas_udf(T.ArrayType(T.FloatType()))
def embed(contents: pd.Series) -> pd.Series:
    import mlflow.deployments

    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_embeddings(batch):
        response = deploy_client.predict(
            endpoint="databricks-bge-large-en", inputs={"input": batch}
        )
        return [e["embedding"] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [
        contents.iloc[i : i + max_batch_size]
        for i in range(0, len(contents), max_batch_size)
    ]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)
