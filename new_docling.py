from docling.document_converter import DocumentConverter

"""
模型下载:docling-tools models download
export DOCLING_ARTIFACTS_PATH="/root/.cache/docling/models"
python new_docling.py
"""
source = "no_git_oic/test_files/流式细胞制备方案.pdf"
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())
