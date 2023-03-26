from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Optional, Type

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseLanguageModel
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
)
from llama_index import QuestionAnswerPrompt
from llama_index.readers.base import BaseReader
from rich.console import Console


class DocsChatBot:
    prompt_helper: PromptHelper
    docs_reader: BaseReader
    llm_predictor: Optional[LLMPredictor] = None
    prompt_template: QuestionAnswerPrompt
    index: GPTVectorStoreIndex
    index_class: Type[GPTVectorStoreIndex]
    idx_local_storage: Optional[Path] = None

    def query(self, question: str) -> str:
        self.index = self.prepare_index()
        resp = self.index.query(question, text_qa_template=self.prompt_template)
        return str(resp)

    def set_index_storage(self, index_storage: Optional[Path] = None):
        self.idx_local_storage = index_storage

    def prepare_index(self) -> GPTVectorStoreIndex:
        if self.idx_local_storage and self.idx_local_storage.exists():
            return self.index_class.load_from_disk(
                self.idx_local_storage.as_posix(), **self.build_index_kwargs()
            )
        return self.create_index()

    def create_index(self):
        docs = self.docs_reader.load_data()
        index = self.index_class(docs, **self.build_index_kwargs())
        if self.idx_local_storage:
            index.save_to_disk(self.idx_local_storage.as_posix())
        return index

    def build_index_kwargs(self):
        return {
            "llm_predictor": self.llm_predictor,
            "prompt_helper": self.prompt_helper,
        }

    def set_reader(self, reader):
        self.docs_reader = reader

    def set_prompt_helper(
        self, max_input_size: int, num_output: int, max_chunk_overlap: int, **kwargs
    ):
        self.prompt_helper = PromptHelper(
            max_input_size, num_output, max_chunk_overlap, **kwargs
        )

    def set_llm_predictor(self, llm: Optional[BaseLanguageModel] = None):
        if not llm:
            self.llm_predictor = None
            return
        self.llm_predictor = LLMPredictor(llm)

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = QuestionAnswerPrompt(prompt_template)

    def set_index_class(self, index_class):
        self.index_class = index_class


class IndexBuilder(metaclass=ABCMeta):
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_prompt_template(self):
        pass

    @abstractmethod
    def set_llm_predictor(self, temperature):
        """define LLM"""
        pass

    @abstractmethod
    def set_prompt_helper(self):
        pass

    @abstractmethod
    def set_docs_reader(
        self, docs_folder: List[Path] = None, docs_path: List[Path] = None, **kwargs
    ):
        pass

    @abstractmethod
    def set_index_class(self):
        pass


class Director:
    def __init__(self, builder: IndexBuilder):
        self.builder = builder

    def construct_text_chatbot(self, docs_path: List[Path]):
        self.builder.reset()
        self.builder.set_docs_reader(docs_path=docs_path)
        self.builder.set_prompt_template()
        self.builder.set_index_class()
        self.builder.set_llm_predictor(0)
        self.builder.set_prompt_helper()


class SimpleDocsIndexBuilder(IndexBuilder):
    chatbot: DocsChatBot

    def reset(self):
        self.chatbot = DocsChatBot()

    def set_prompt_template(self):
        prompt_tmpl = (
            "Providing you context info below.\n"
            "---------------------\n"
            "{context_str}"
            "---------------------\n"
            "Given this info, please answer the question with refer url: {query_str}\n"
        )
        self.chatbot.set_prompt_template(prompt_tmpl)

    def set_llm_predictor(self, temperature: float = 0):
        """define LLM"""
        self.chatbot.set_llm_predictor(ChatOpenAI(temperature=0, client=None))

    def set_prompt_helper(self):
        max_input_size = 4096
        num_output = 512
        max_chunk_overlap = 20
        self.chatbot.set_prompt_helper(max_input_size, num_output, max_chunk_overlap)

    def set_docs_reader(
        self, docs_folder: List[Path] = None, docs_path: List[Path] = None, **kwargs
    ):
        pre_kwargs = {
            "input_dir": docs_folder,
            "input_files": docs_path,
        }
        self.chatbot.set_reader(SimpleDirectoryReader(**pre_kwargs, **kwargs))

    def set_index_class(self):
        self.chatbot.set_index_class(GPTSimpleVectorIndex)


if __name__ == "__main__":
    console = Console()
    builder = SimpleDocsIndexBuilder()
    director = Director(builder)
    director.construct_text_chatbot([Path("data_simple.txt")])

    count = 1
    while True:
        console.rule(f"Quiz [{count}]")
        quiz = console.input("[blue underline]Question: ")
        with console.status("Querying...", spinner="dots3"):
            response = builder.chatbot.query(quiz)
            console.print(f"[green]{response}")
        count += 1
