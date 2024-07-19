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

#
# def read_with_pypdf(data):
#     reader = PdfReader(io.BytesIO(data))
#     text = ""
#     for page_num in range(len(reader.pages))[0:1]:
#         page_obj = reader.pages[page_num]
#         _text = page_obj.extract_text(extraction_mode="plain")
#         # _text = re.sub(r'\n\n', '\n', _text)
#         # _text = re.sub(r' ?\.', '.', _text)
#         text += _text
#     return text

#
# def read_with_pdfminer(data):
#     text = extract_text(io.BytesIO(data))
#     # text = re.sub(r'\n', '', text)
#     text = re.sub(r' ?\.', '.', text)
#     text = re.sub("", '.', text)
#     return text

# def read_with_unstructured(data):
#   sections = partition(file=io.BytesIO(data))
#   def clean_section(txt):
#         txt = re.sub(r'\n', '', txt)
#         return re.sub(r' ?\.', '.', txt)
#   # Default split is by section of document
#   # concatenate them all together because we want to split by sentence instead.
#   return "\n".join([clean_section(s.text) for s in sections])


@F.pandas_udf(T.ArrayType(T.StringType()))
# @F.pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:

    # TODO: test outside of the function

    def read_with_pdfminer(data):
        text = extract_text(io.BytesIO(data))
        # text = re.sub(r'\n', '', text)
        text = re.sub(r' ?\.', '.', text)
        text = re.sub("", '.', text)
        return text

    #set llama2 as tokenizer
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", cache_dir="/Workspace/tmp/huggingface/")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = read_with_pdfminer(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)
