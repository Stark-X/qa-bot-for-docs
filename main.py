import logging
import sys

import chromadb
from llama_index import GPTChromaIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from pathlib import Path
from reader.markdown import MakrdownReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# define LLM
llm_predictor = LLMPredictor(OpenAI(
    temperature=0, model_name="gpt-3.5-turbo",  client=None))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)


# in-memory store
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection("test")

reader = MakrdownReader("data")
docs = reader.load_data()

#  index = GPTChromaIndex(docs, chroma_collection)
if (index_store := Path("index.json")).exists():
    index = GPTSimpleVectorIndex.load_from_disk(index_store.as_posix())
else:
    index = GPTSimpleVectorIndex(
        docs, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk(index_store.as_posix())

while True:
    quiz = input("Quiestion:")
    response = index.query(quiz)
    print(response)
    print(f"Token usage: {index.llm_predictor.last_token_usage}")
