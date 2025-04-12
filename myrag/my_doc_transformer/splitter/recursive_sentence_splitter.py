import os
from typing import Optional, List
from dotenv import load_dotenv
import logging
from termcolor import colored
from datetime import datetime
import sys
from haystack import Document
import spacy
from tqdm import tqdm
from copy import deepcopy

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

"""
https://spacy.io/models/zh
uv pip install no_git_oic/whl/zh_core_web_sm-3.8.0-py3-none-any.whl
uv pip install no_git_oic/whl/en_core_web_sm-3.8.0-py3-none-any.whl
uv pip install no_git_oic/whl/zh_core_web_trf-3.8.0-py3-none-any.whl
uv pip install no_git_oic/whl/en_core_web_trf-3.8.0-py3-none-any.whl

uv pip install /path/to/en_core_web_lg-3.8.0.tar.gz
uv pip install /path/to/zh_core_web_lg-3.8.0.tar.gz
"""

LANG2MODELNAME = {
    "en": "en_core_web_trf",
    "zh": "zh_core_web_trf",
    "en_sm": "en_core_web_sm",
    "zh_sm": "zh_core_web_sm",
}


class RecursiveSentenceSplitter:
    NAME = "RecursiveSentenceSplitter"

    def __init__(
        self,
        lang: str = "zh",
        *,
        nlp_max_len: int = 400000,
        num_parallel: int = 8,
        chunk_size: int = 12,
        chunk_overlap: int = 4,
    ):
        """
        Args:
            lang (str):  "zh" for Chinese. "en" for English en_sm zh_sm
            chunk_size (int): number of sentences per chunk.
            chunk_overlap (int): number of sentence overlap between two continuous chunks.
        """
        self._chunk_overlap = chunk_overlap
        self._chunk_size: int = chunk_size
        self._stride: int = self._chunk_size - self._chunk_overlap
        self._num_parallel: int = num_parallel

        self._load_model(lang, nlp_max_len)

    def _load_model(self, language: str, nlp_max_len: int) -> None:
        assert (
            language in LANG2MODELNAME
        ), f"Spacy model not specified for language: {language}."

        model_name = LANG2MODELNAME[language]
        try:
            self._nlp = spacy.load(model_name)
        except:
            raise ValueError(
                f"Spacy model not found for language: {language}. Please install it first."
            )
        self._nlp.max_length = nlp_max_len

    def _nlp_doc_to_texts(self, doc: spacy.tokens.Doc) -> List[str]:
        sents = [sent.text.strip() for sent in doc.sents]
        sents = [sent for sent in sents if len(sent) > 0]

        segments: List[str] = []
        for i in range(0, len(sents), self._stride):
            segment = " ".join(sents[i : i + self._chunk_size])
            segments.append(segment)
            if i + self._chunk_size >= len(sents):
                break

        return segments

    def _split_text(self, text: str) -> List[str]:
        """
        仅测试块的分割
        """
        doc = self._nlp(text)
        segments = self._nlp_doc_to_texts(doc)
        return segments

    def create_documents(self, document: Document) -> List[Document]:
        text = document.content
        meta = document.meta if hasattr(document, "meta") else {}
        documents = []
        num_workers = min(1, self._num_parallel)

        pbar = tqdm(total=1, desc="Splitting document by sentences")

        # 处理文档
        for doc in self._nlp.pipe([text], n_process=num_workers, batch_size=32):
            segments = self._nlp_doc_to_texts(doc)
            for segment in segments:
                documents.append(
                    Document(
                        content=segment,
                        meta=deepcopy(meta),
                    )
                )
            pbar.update(1)
        pbar.close()

        return documents


if __name__ == "__main__":
    """
    uv run myrag/my_doc_transformer/splitter/recursive_sentence_splitter.py
    """

    file_path = "no_git_oic/test_files/AI应用培训01_V2.1_0228.pptx"
    converter = get_loader(file_path)
    results = converter.run(
        sources=[file_path],
        meta={"date_added": datetime.now().isoformat()},
    )
    documents = results["documents"]

    splitter = RecursiveSentenceSplitter("zh", chunk_size=12, chunk_overlap=4)
    segments = splitter._split_text(documents[0].content)
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}:")
        logger.info(colored(f"{segment}", "green"))

    documents = splitter.create_documents(documents[0])
    for i, doc in enumerate(documents):
        print(f"\n文档 {i+1}:")
        logger.info(colored(f"内容: {doc.content}", "green"))
        logger.info(colored(f"元数据: {doc.meta}", "light_cyan"))
