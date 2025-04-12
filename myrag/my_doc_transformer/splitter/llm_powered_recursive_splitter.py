from typing import List, Tuple
from tqdm import tqdm
import os
from dotenv import load_dotenv
import logging
from termcolor import colored
import sys
from datetime import datetime
from haystack import Document

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
from myrag.my_doc_transformer.splitter.recursive_sentence_splitter import (
    RecursiveSentenceSplitter,
)
from myrag.llm_client.base import BaseLLMClient
from myrag.my_prompt.prompts import *
from z_utils.hash import compute_mdhash_id


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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_first_chunk_summary(self, text: str) -> str:
        chunks = self._base_splitter.split_text(text)
        # logger.info(colored(f"len(chunks):{len(chunks)}", "green"))
        # 直接获取分割后的第一个chunk 可能会丢失原文中chunk前的一些空白字符或特殊格式
        first_chunk_start_pos = text.find(chunks[0])
        text_for_summary = text[: first_chunk_start_pos + len(chunks[0])]

        messages = [
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": "Tell me the main topics of this article in sevaral sentences , only return the result.\n\nText:\n```"
                + text_for_summary
                + "\n```",
            },
        ]
        response = self._llm_client.generate_content_with_messages(
            messages, **self._llm_config
        )
        return response

    def _get_chunk_summary(self, text: str, last_chunk_summary: str) -> str:
        messages = [
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": (
                    "Help me generate summaries, only return the answer.\n"
                    f"Previous summary:\n```{last_chunk_summary}```\n"
                    f"Text:\n```{text}```\n"
                ),
            },
        ]
        response = self._llm_client.generate_content_with_messages(
            messages, **self._llm_config
        )
        return response

    def split_documents(self, document: Document, **kwargs) -> List[Document]:
        """
        1. 先使用 self._base_splitter 初步分割器初步分割 获得 List[Document]==>base_document_list
        2. 然后循环读取base_document_list合并其content 到 chunk_size 大小,但是不能超过他,然后使用 LLM 进行首块摘要总结 start_summary
        3. 重新分割文档 开始循环读取base_document_list合并其content到 chunk_size 大小,但是不能超过他,
        此函数还有很多问题,算了,就简单切分好了

        Document是content和meta,id,而meta里面包括summary,chunk_index,file_path,date_added
        """
        ret_docs: List[Document] = []
        base_document_list = self._base_splitter.create_documents(document)

        if not base_document_list:
            return ret_docs

        # 1. 获取文章主旨
        for doc in tqdm(base_document_list, desc="LLM Splitting Documents"):
            current_chunk = ""
            text = doc.content
            if len(current_chunk) + len(text) <= self.chunk_size:
                current_chunk += text
            start_summary = self._get_first_chunk_summary(current_chunk)
            break

        # 2. 重新分割文档,并且获取每块文档的摘要,以及他的 chunk_index(指在全文中当前chunk文档开始的坐标)
        while True:
            for doc in tqdm(base_document_list, desc="LLM Splitting Documents"):
                current_chunk = ""
                last_chunk_summary = ""
                text = doc.content

                if len(current_chunk) + len(text) <= self.chunk_size:
                    current_chunk += text
                else:
                    chunk_summary, chunk_index = self._resplit_summary_chunk(
                        start_summary, last_chunk_summary, current_chunk
                    )
                    ret_docs.append(
                        Document(
                            id=compute_mdhash_id(current_chunk),
                            content=current_chunk,
                            meta={
                                "summary": chunk_summary,
                                "chunk_index": chunk_index,
                                "file_path": document.meta["file_path"],
                                "date_added": document.meta["date_added"],
                            },
                        )
                    )

        return ret_docs


if __name__ == "__main__":
    """
    uv run myrag/my_doc_transformer/splitter/llm_powered_recursive_splitter.py
    """
    llm_client = BaseLLMClient()
    file_path = "no_git_oic/test_files/AI应用培训01_V2.1_0228.pptx"
    # file_path = "no_git_oic/test_files/linux环境安装代理VPN的步骤.txt"

    converter = get_loader(file_path)
    results = converter.run(
        sources=[file_path],
        meta={"date_added": datetime.now().isoformat()},
    )
    documents = results["documents"]

    llm_config = {"re_run": False, "model": "gemma3:27b"}
    splitter = LLMPoweredRecursiveSplitter(llm_client, llm_config=llm_config)

    split_doc_list = splitter.split_documents(documents[0])
    logger.info(colored(f"{split_doc_list}", "green"))
