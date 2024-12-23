from floki.types.document import Document
from floki.document.reader.base import ReaderBase
from typing import List
from pathlib import Path

class PyPDFReader(ReaderBase):
    """
    Reader for PDF documents using PyPDF.
    """

    def load(self, file_path: Path) -> List[Document]:
        """
        Load content from a PDF file using PyPDF.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            List[Document]: A list of Document objects.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF library is not installed. Install it using `pip install pypdf`."
            )

        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        documents = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            metadata = {
                "file_path": str(file_path),
                "page_number": page_num + 1,
                "total_pages": total_pages,
            }
            documents.append(Document(text=text.strip(), metadata=metadata))

        return documents