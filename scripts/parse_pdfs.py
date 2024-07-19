# from llmsherpa.readers import LayoutPDFReader
#
# llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# pdf_url = "./analyst-associate.pdf"  # also allowed is a file path e.g. /home/downloads/xyz.pdf
# pdf_reader = LayoutPDFReader(llmsherpa_api_url)
# doc = pdf_reader.read_pdf(pdf_url)
#

from pypdf import PdfReader
import time
import io
import re
# t0 = time.time()
from unstructured.partition.auto import partition
# print(f"imported unstructured in {time.time()-t0:5.2f}")
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer
from llama_index.core import Document
from llama_index.core.utils import set_global_tokenizer
from llama_index.core.langchain_helpers.text_splitter import SentenceSplitter

filepath = "./analyst-associate.pdf"

# text = extract_text("./analyst-associate.pdf")


def read_with_pypdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page_num in range(len(reader.pages))[0:1]:
        page_obj = reader.pages[page_num]
        _text = page_obj.extract_text(extraction_mode="plain")
        # _text = re.sub(r'\n\n', '\n', _text)
        # _text = re.sub(r' ?\.', '.', _text)
        text += _text
    return text


def read_with_pdfminer(filepath):
    text = extract_text(filepath)
    # text = re.sub(r'\n', '', text)
    text = re.sub(r' ?\.', '.', text)
    text = re.sub("", '.', text)
    return text

def read_with_unstructured(filepath):
  # Read files and extract the values with unstructured

  with open(filepath, "rb") as fp:
      data = fp.read()

  sections = partition(file=io.BytesIO(data))
  # sections = partition(file=filepath)
  def clean_section(txt):
        txt = re.sub(r'\n', '', txt)
        return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document
  # concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections])


def build_chunks(text):
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    # #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    return [n.text for n in nodes]
    #
    # for x in batch_iter:
    #     yield x.apply(extract_and_split)


t0 = time.time()
text_pypdf = read_with_pypdf(filepath)
print(f"Read with pypdf in {time.time()-t0:5.2f}")

t0 = time.time()
text_pdfminer = read_with_pdfminer(filepath)
print(f"Read with pdfminer {time.time()-t0:5.2f}")
with open("pdfminer.txt", "w") as fp:
    fp.write(text_pdfminer)


t0 = time.time()
text_unstructured = read_with_unstructured(filepath)
print(f"Read with pdfminer {time.time()-t0:5.2f}")
with open("unstructured.txt", "w") as fp:
    fp.write(text_unstructured)

t0 = time.time()
chunks = build_chunks(text_unstructured)
print(f"Build chunks {time.time()-t0:5.2f}")
with open("chunks.txt", "w") as fp:
    for c in chunks:
        fp.write(c)


#
#
# import re
# import io
# import pandas as pd
# import pyspark.sql.functions as F
# import pyspark.sql.functions as T
#
# from llama_index.core import Document
# from llama_index.core.utils import set_global_tokenizer
# from llama_index.core.langchain_helpers.text_splitter import SentenceSplitter
# from transformers import AutoTokenizer
# from typing import Iterator
# from unstructured.partition.auto import partition
#
# # spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
#
#
# @F.pandas_udf(T.StringType())
# def path_to_type(s: pd.Series) -> pd.Series:
#     return F.split(s, )
#
#
# def extract_doc_text(x: bytes) -> str:
#   # Read files and extract the values with unstructured
#   sections = partition(file=io.BytesIO(x))
#   def clean_section(txt):
#     txt = re.sub(r'\n', '', txt)
#     return re.sub(r' ?\.', '.', txt)
#   # Default split is by section of document
#   # concatenate them all together because we want to split by sentence instead.
#   return "\n".join([clean_section(s.text) for s in sections])
#
# #
# # @F.pandas_udf(T.ArrayType(T.StringType()))
# # # @F.pandas_udf("array<string>")
# # def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
# #     #set llama2 as tokenizer
# #     # set_global_tokenizer(
# #     #   AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
# #     # )
# #     #Sentence splitter from llama_index to split on sentences
# #     splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
# #     def extract_and_split(b):
# #       txt = extract_doc_text(b)
# #       nodes = splitter.get_nodes_from_documents([Document(text=txt)])
# #       return [n.text for n in nodes]
# #
# #     for x in batch_iter:
# #         yield x.apply(extract_and_split)
# #
#
# with open(f"./analyst-associate.pdf", mode="rb") as pdf:
#   doc = extract_doc_text(pdf.read())
#   print(doc)
