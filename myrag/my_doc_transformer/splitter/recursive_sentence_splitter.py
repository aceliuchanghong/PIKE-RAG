import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging
from termcolor import colored
from datetime import datetime
import sys
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

from myrag.my_loader.common import DocumentType

"""
https://spacy.io/models/zh
pip install no_git_oic/spacy/zh_core_web_sm-3.8.0-py3-none-any.whl
pip install no_git_oic/spacy/en_core_web_sm-3.8.0-py3-none-any.whl

pip install /path/to/en_core_web_lg-3.8.0.tar.gz
pip install /path/to/zh_core_web_lg-3.8.0.tar.gz
"""
if __name__ == "__main__":
    import spacy

    """
    uv run myrag/my_doc_transformer/splitter/recursive_sentence_splitter.py
    """
    nlp_sm = spacy.load("zh_core_web_sm")
    doc_sm = nlp_sm("这是一个测试句子。")
    print([(token.text, token.pos_, token.dep_) for token in doc_sm])
