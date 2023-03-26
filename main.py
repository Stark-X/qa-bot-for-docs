from pathlib import Path

from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
)
from llama_index import QuestionAnswerPrompt
from rich.console import Console


# define LLM
llm_predictor = LLMPredictor(ChatOpenAI(temperature=0,  client=None))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)


reader = SimpleDirectoryReader(input_files=["data.txt"])
docs = reader.load_data()

QA_PROMPT_TMPL = (
    "Providing you context info below.\n"
    "---------------------\n"
    "{context_str}"
    "---------------------\n"
    "Given this info, please answer the question with refer url: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
#  index = GPTChromaIndex(docs, chroma_collection)
if (index_store := Path("index.json")).exists():
    index = GPTSimpleVectorIndex.load_from_disk(index_store.as_posix(
    ), llm_predictor=llm_predictor, prompt_helper=prompt_helper)
else:
    index = GPTSimpleVectorIndex(
        docs, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk(index_store.as_posix())


console = Console()

count = 1
while True:
    console.rule(f"Quiz [{count}]")
    quiz = console.input("[blue underline]Quiestion: ")
    with console.status("Querying...", spinner="dots3"):
        response = index.query(quiz, text_qa_template=QA_PROMPT)
        console.print(f"[green]{response}")
    count += 1
