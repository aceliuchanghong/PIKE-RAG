from enum import Enum


class DocumentType(Enum):
    csv = ["csv"]
    excel = ["xlsx"]
    markdown = ["md"]
    text = ["txt"]
    word = ["docx"]
    ppt = ["pptx"]
    pdf = ["pdf"]
