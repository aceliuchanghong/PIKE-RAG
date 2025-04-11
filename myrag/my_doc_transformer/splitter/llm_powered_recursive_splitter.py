from copy import deepcopy
from typing import Callable, Iterable, List, Tuple
from tqdm import tqdm
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
from openai import OpenAI


load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    ),
)

from myrag.my_loader.utils import get_loader
from datetime import datetime
from haystack import Document
from myrag.my_doc_transformer.splitter.recursive_sentence_splitter import (
    RecursiveSentenceSplitter,
)
from myrag.llm_client.base import BaseLLMClient
from myrag.my_prompt.prompts import *


class LLMPoweredRecursiveSplitter:
    NAME = "LLMPoweredRecursiveSplitter"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        llm_config: dict = {},
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ) -> None:

        self._base_splitter = RecursiveSentenceSplitter()
        self._llm_client = llm_client
        self._llm_config = llm_config

    def _get_first_chunk_summary(self, text: str) -> str:
        chunks = self._base_splitter.split_text(text)
        # 直接获取分割后的第一个chunk 可能会丢失原文中chunk前的一些空白字符或特殊格式
        first_chunk_start_pos = text.find(chunks[0])
        text_for_summary = text[: first_chunk_start_pos + len(chunks[0])]

        messages = [
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": "Help me summarize the text, only return the summary.\n\nText:\n"
                + text_for_summary,
            },
        ]
        response = self._llm_client.generate_content_with_messages(
            messages, **self._llm_config
        )
        return response

    def _resplit_chunk_and_generate_summary(
        self,
        text: str,
        chunks: List[str],
        chunk_summary: str,
        **kwargs,
    ) -> Tuple[str, str, str, str]:
        assert (
            len(chunks) >= 2
        ), f"When calling this function, input chunks length should be no less than 2!"
        text_to_resplit = text[: len(chunks[0]) + len(chunks[1])]

        kwargs["summary"] = chunk_summary
        messages = self._chunk_resplit_protocol.process_input(
            content=text_to_resplit, **kwargs
        )
        response = self._llm_client.generate_content_with_messages(messages=messages)
        return self._chunk_resplit_protocol.parse_output(content=response, **kwargs)

    def _get_last_chunk_summary(self, chunk: str, chunk_summary: str, **kwargs) -> str:
        kwargs["summary"] = chunk_summary
        messages = self._last_chunk_summary_protocol.process_input(
            content=chunk, **kwargs
        )
        response = self._llm_client.generate_content_with_messages(messages=messages)
        return self._last_chunk_summary_protocol.parse_output(
            content=response, **kwargs
        )

    def split_text(self, text: str, metadata: dict) -> List[str]:
        docs = self.create_documents(texts=[text], metadatas=[metadata])
        return [doc.page_content for doc in docs]

    def create_documents(
        self, texts: List[str], metadatas: List[dict], **kwargs
    ) -> List[Document]:
        if len(texts) != len(metadatas):
            raise ValueError(
                f"Input texts and metadatas should have same length, "
                f"{len(texts)} texts but {len(metadatas)} metadatas are given."
            )

        ret_docs: List[Document] = []
        for text, metadata in zip(texts, metadatas):
            ret_docs.extend(
                self.split_documents(
                    [Document(page_content=text, metadata=metadata)], **kwargs
                )
            )
        return ret_docs

    def split_documents(
        self, documents: Iterable[Document], **kwargs
    ) -> List[Document]:
        ret_docs: List[Document] = []
        for doc in tqdm(documents, desc="Splitting Documents"):
            text = doc.page_content
            metadata = doc.metadata

            text = text.strip()
            chunk_summary = self._get_first_chunk_summary(text, **metadata)
            chunks = self._base_splitter.split_text(text)
            while True:
                if len(chunks) == 1:
                    chunk_summary = self._get_last_chunk_summary(
                        chunks[0], chunk_summary, **metadata
                    )
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(
                        Document(page_content=chunks[0], metadata=chunk_meta)
                    )
                    break

                else:
                    chunk, chunk_summary, next_summary, dropped_len = (
                        self._resplit_chunk_and_generate_summary(
                            text,
                            chunks,
                            chunk_summary,
                            **metadata,
                        )
                    )

                    if len(chunk) == 0:
                        chunk_summary = next_summary
                        chunks = [chunks[0] + chunks[1]] + chunks[2:]
                        continue

                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(Document(page_content=chunk, metadata=chunk_meta))

                    text = text[dropped_len:].strip()
                    chunk_summary = next_summary
                    chunks = self._base_splitter.split_text(text)

        return ret_docs


if __name__ == "__main__":
    """
    uv run myrag/my_doc_transformer/splitter/llm_powered_recursive_splitter.py
    """
    llm_client = BaseLLMClient()
    file_path = "no_git_oic/test_files/linux环境安装代理VPN的步骤.txt"
    converter = get_loader(file_path)
    results = converter.run(
        sources=[file_path],
        meta={"date_added": datetime.now().isoformat()},
    )
    documents = results["documents"]
    logger.info(colored(f"documents:{documents}", "green"))

    llm_config = {"re_run": False, "model": "gemma3:27b"}
    splitter = LLMPoweredRecursiveSplitter(llm_client, llm_config=llm_config)
    first_chunk_summary = splitter._get_first_chunk_summary(documents[0].content)
    logger.info(colored(f"{first_chunk_summary}", "green"))
