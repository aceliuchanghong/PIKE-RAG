from copy import deepcopy
from typing import Iterable, List, Tuple
from tqdm import tqdm
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys


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
        logger.info(colored(f"len(chunks):{len(chunks)}", "green"))
        # 直接获取分割后的第一个chunk 可能会丢失原文中chunk前的一些空白字符或特殊格式
        first_chunk_start_pos = text.find(chunks[0])
        text_for_summary = text[: first_chunk_start_pos + len(chunks[0])]

        messages = [
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": "Tell me the main topics of this article, only return the description.\n\nText:\n```"
                + text_for_summary
                + "\n```",
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
        assert len(chunks) >= 2, "Input chunks length should be no less than 2!"

        text_to_process = chunks[0] + chunks[1]

        messages = [
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": (
                    "Help me split this text into two parts and generate summaries.\n"
                    f"Previous summary:\n```{chunk_summary}```\n"
                    f"Text to split:\n```{text_to_process}```\n"
                    "Return format:\n"
                    "First part:\n<first part content>\n"
                    "First part summary:\n<summary1>\n"
                    "Second part summary:\n<summary2>"
                ),
            },
        ]

        response = self._llm_client.generate_content_with_messages(
            messages, **self._llm_config
        )

        try:
            # Parse the response
            parts = response.split("First part:\n", 1)[1]
            first_part, summaries = parts.split("First part summary:\n", 1)
            first_summary, second_summary = summaries.split("Second part summary:\n", 1)

            first_part = first_part.strip()
            first_summary = first_summary.strip()
            second_summary = second_summary.strip()

            return first_part, first_summary, second_summary, len(first_part)
        except:
            # Return empty result if parsing fails
            return "", chunk_summary, chunk_summary, 0

    def _get_last_chunk_summary(self, chunk: str, chunk_summary: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": "Help me summarize the text, only return the summary.\nLast_chunk_summary:\n```"
                + chunk_summary
                + "```\nText:\n```"
                + chunk
                + "\n```",
            },
        ]
        response = self._llm_client.generate_content_with_messages(
            messages, **self._llm_config
        )
        return response

    def split_text(self, text: str, meta: dict) -> List[str]:
        docs = self.create_documents(texts=[text], metas=[meta])
        return [doc.content for doc in docs]

    def create_documents(
        self, texts: List[str], metas: List[dict], **kwargs
    ) -> List[Document]:
        if len(texts) != len(metas):
            raise ValueError(
                f"Input texts and metadatas should have same length, "
                f"{len(texts)} texts but {len(metas)} metadatas are given."
            )

        ret_docs: List[Document] = []
        for text, metadata in zip(texts, metas):
            ret_docs.extend(
                self.split_documents([Document(content=text, meta=metadata)], **kwargs)
            )
        return ret_docs

    def split_documents(
        self, documents: Iterable[Document], **kwargs
    ) -> List[Document]:
        ret_docs: List[Document] = []
        for doc in tqdm(documents, desc="Splitting Documents"):
            text = doc.content
            metadata = doc.meta

            text = text.strip()
            chunk_summary = self._get_first_chunk_summary(text)
            chunks = self._base_splitter.split_text(text)
            while True:
                if len(chunks) == 1:
                    logger.info(colored(f"QBQ", "green"))
                    chunk_summary = self._get_last_chunk_summary(
                        chunks[0], chunk_summary
                    )
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(Document(content=chunks[0], meta=chunk_meta))
                    break

                else:
                    logger.info(colored(f"QAQ", "green"))
                    # 重新分割并处理当前块
                    chunk, chunk_summary, next_summary, dropped_len = (
                        self._resplit_chunk_and_generate_summary(
                            text,
                            chunks,
                            chunk_summary,
                            **metadata,
                        )
                    )

                    if len(chunk) == 0:
                        # 分割失败，合并块重试
                        chunk_summary = next_summary
                        chunks = [chunks[0] + chunks[1]] + chunks[2:]
                        continue

                    # 添加处理好的块
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(Document(content=chunk, meta=chunk_meta))
                    # 更新剩余文本信息
                    text = text[dropped_len:].strip()
                    chunk_summary = next_summary
                    chunks = self._base_splitter.split_text(text)

        return ret_docs


if __name__ == "__main__":
    """
    uv run myrag/my_doc_transformer/splitter/llm_powered_recursive_splitter.py
    """
    llm_client = BaseLLMClient()
    file_path = "no_git_oic/test_files/AI应用培训01_V2.1_0228.pptx"
    converter = get_loader(file_path)
    results = converter.run(
        sources=[file_path],
        meta={"date_added": datetime.now().isoformat()},
    )
    documents = results["documents"]
    # logger.info(colored(f"documents:{documents}", "green"))

    llm_config = {"re_run": False, "model": "gemma3:27b"}
    splitter = LLMPoweredRecursiveSplitter(llm_client, llm_config=llm_config)
    first_chunk_summary = splitter._get_first_chunk_summary(documents[0].content)
    logger.info(colored(f"first_chunk_summary:{first_chunk_summary}", "green"))

    texts = documents[0].content
    meta = documents[0].meta
    new_doc_list = splitter.split_text(texts=texts, metadatas=meta)
    logger.info(colored(f"new_doc_list:{new_doc_list},{len(new_doc_list)}", "green"))
