from typing import List, Iterable, Dict, Any, Tuple
from copy import deepcopy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI


class LLMSplitter:
    def __init__(self, llm=None, chunk_size: int = 4000, chunk_overlap: int = 200):
        self.llm = llm or ChatOpenAI()
        self._base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _get_initial_summary(self, text: str) -> str:
        """Get initial summary for the document."""
        prompt = (
            f"Please provide a brief summary of the following text:\n{text[:2000]}..."
        )
        return self.llm.predict(prompt)

    def _get_chunk_summary(self, chunk: str, prev_summary: str) -> str:
        """Generate summary for a chunk considering previous summary."""
        prompt = f"Previous summary: {prev_summary}\nPlease provide a summary for this new chunk:\n{chunk}"
        return self.llm.predict(prompt)

    def _resplit_chunks(
        self, text: str, chunks: List[str], summary: str
    ) -> Tuple[str, str, str, int]:
        """Resplit first two chunks and generate summaries."""
        text_to_resplit = chunks[0] + chunks[1]
        prompt = f"Split the following text into two coherent parts:\n{text_to_resplit}"
        split_point = len(chunks[0])  # Default to original split point

        try:
            # Try to get a better split point from LLM
            response = self.llm.predict(prompt)
            if "|" in response:
                split_point = text_to_resplit.find(response.split("|")[1].strip())
        except:
            pass

        if split_point > 0:
            chunk = text_to_resplit[:split_point]
            chunk_summary = self._get_chunk_summary(chunk, summary)
            next_summary = self._get_chunk_summary(
                text_to_resplit[split_point:], chunk_summary
            )
            return chunk, chunk_summary, next_summary, split_point

        return chunks[0], summary, summary, len(chunks[0])

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents into chunks with summaries."""
        result_docs: List[Document] = []

        for doc in documents:
            text = doc.page_content.strip()
            metadata = doc.metadata

            # Initial document processing
            chunk_summary = self._get_initial_summary(text)
            chunks = self._base_splitter.split_text(text)

            while True:
                if len(chunks) == 1:
                    # Process last chunk
                    chunk_summary = self._get_chunk_summary(chunks[0], chunk_summary)
                    chunk_meta = deepcopy(metadata)
                    chunk_meta["summary"] = chunk_summary
                    result_docs.append(
                        Document(page_content=chunks[0], metadata=chunk_meta)
                    )
                    break
                else:
                    # Process and resplit chunks
                    chunk, chunk_summary, next_summary, dropped_len = (
                        self._resplit_chunks(text, chunks, chunk_summary)
                    )

                    if len(chunk) == 0:
                        # Merge chunks if split failed
                        chunks = [chunks[0] + chunks[1]] + chunks[2:]
                        continue

                    # Add processed chunk
                    chunk_meta = deepcopy(metadata)
                    chunk_meta["summary"] = chunk_summary
                    result_docs.append(
                        Document(page_content=chunk, metadata=chunk_meta)
                    )

                    # Update remaining text and chunks
                    text = text[dropped_len:].strip()
                    chunk_summary = next_summary
                    chunks = self._base_splitter.split_text(text)

        return result_docs
