"""Markdown files reader class.

Parase all markdown files with only text content.
Empty files will be ignored.

"""
import os
from pathlib import Path
from typing import Any, List

from langchain.docstore.document import Document as LCDocument

from llama_index.readers.base import BaseReader
from llama_index.readers.file.markdown_parser import MarkdownParser
from llama_index.readers.schema.base import Document


class MakrdownReader(BaseReader):
    """Utilities for loading data from markdown folder, e.g. Obsidian Vault.

    Args:
        input_dir (str): Path to the vault.

    """

    def __init__(self, input_dir: str):
        """Init params."""
        self.input_dir = Path(input_dir)

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        docs: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            # ignore hidden folders
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if not filename.endswith(".md"):
                    continue
                filepath = Path(dirpath) / filename
                # prevent "out of index" exception
                if filepath.stat().st_size == 0:
                    continue
                content = MarkdownParser().parse_file(filepath)
                docs.extend(content)
        return [Document(d) for d in docs]

    def load_langchain_documents(self, **load_kwargs: Any) -> List[LCDocument]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]
