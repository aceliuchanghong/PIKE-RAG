import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging
from termcolor import colored
from datetime import datetime
from haystack.components.converters import (
    MarkdownToDocument,
    CSVToDocument,
    DOCXToDocument,
    PPTXToDocument,
    PDFMinerToDocument,
    TextFileToDocument,
    XLSXToDocument,
)
from haystack.components.converters.docx import DOCXTableFormat

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

from myrag.my_loader.common import DocumentType


def is_pdf_readable(file_path: str) -> bool:
    """
    检查 PDF 文件是否包含可读文本。
    :param file_path: PDF 文件路径
    :return: 如果 PDF 包含可读文本，返回 True 否则返回 False
    """
    import pdfplumber

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    return True
        return False
    except Exception as e:
        print(f"Error while checking PDF readability: {e}")
        return False


def infer_file_type(file_path: str) -> Optional[DocumentType]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件路径不存在: {file_path}")

    file_extension = Path(file_path).suffix[1:].lower()

    for doc_type in DocumentType:
        if file_extension in doc_type.value:
            return doc_type

    supported_types = ", ".join(
        [f".{ext}" for doc_type in DocumentType for ext in doc_type.value]
    )
    logger.error(
        colored(f"不支持的文件类型。仅支持以下文件类型: {supported_types}", "red")
    )
    return None


def get_loader(file_path: str):
    inferred_file_type = infer_file_type(file_path)

    if inferred_file_type == DocumentType.csv:
        return CSVToDocument()
    elif inferred_file_type == DocumentType.excel:
        return XLSXToDocument()
    elif inferred_file_type == DocumentType.markdown:
        return MarkdownToDocument()
    elif inferred_file_type == DocumentType.text:
        return TextFileToDocument()
    elif inferred_file_type == DocumentType.word:
        return DOCXToDocument()
    elif inferred_file_type == DocumentType.ppt:
        return PPTXToDocument()
    elif inferred_file_type == DocumentType.pdf:
        if not is_pdf_readable(file_path):
            print("不支持包含图像的 PDF 文件，请提供可读文本的 PDF。")
            return None
        return PDFMinerToDocument()
    else:
        print(f"Converter for type {inferred_file_type} not defined.")
        return None


if __name__ == "__main__":
    """
    uv run myrag/my_loader/utils.py
    """
    file_path = "no_git_oic/test_files/IMDB-Movie-Data.csv"
    file_path = "no_git_oic/test_files/8af0caa2d85618671dbcd392771dc086ef6aba991f11e40b124514f11deed8e9/流式细胞制备方案.pdf"
    file_path = "no_git_oic/test_files/RAI_TRANSPARENCY.md"
    file_path = "no_git_oic/test_files/AI应用培训01_V2.1_0228.pptx"
    file_path = "no_git_oic/test_files/test_excel.xlsx"
    file_path = "no_git_oic/test_files/三国演义.docx"
    file_path = "no_git_oic/test_files/linux环境安装代理VPN的步骤.txt"

    doc_type = infer_file_type(file_path)
    if doc_type:
        print(f"文件类型: {doc_type.name}")

        converter = get_loader(file_path)
        results = converter.run(
            sources=[file_path],
            meta={"date_added": datetime.now().isoformat()},
        )
        documents = results["documents"]
        print(documents)
        print(documents[0].content)
    else:
        print("无法识别文件类型")
