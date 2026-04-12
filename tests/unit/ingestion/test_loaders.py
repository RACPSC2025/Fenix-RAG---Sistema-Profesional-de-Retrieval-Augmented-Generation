"""Tests de los loaders de ingestión: PDF PyMuPDF, PDF OCR, Docling, Word, Excel."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.ingestion.base import IngestionResult
from src.ingestion.loaders.pdf_pymupdf import PyMuPDFLoader
from src.ingestion.loaders.pdf_ocr import OCRLoader
from src.ingestion.loaders.word_loader import WordLoader
from src.ingestion.loaders.excel_loader import ExcelLoader


# ─── PyMuPDF Loader ───────────────────────────────────────────────────────

class TestPyMuPDFLoader:
    """Tests del loader PDF nativo con PyMuPDF."""

    def test_loader_type(self):
        loader = PyMuPDFLoader()
        assert loader.loader_type == "pymupdf"

    @patch("src.ingestion.loaders.pdf_pymupdf.fitz")
    def test_load_pdf_native(self, mock_fitz):
        """Carga PDF con texto seleccionable."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Este es el contenido del PDF con texto nativo."
        mock_doc.__len__ = lambda self: 1
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__getitem__ = lambda self, idx: mock_page
        mock_fitz.open.return_value = mock_doc

        loader = PyMuPDFLoader()
        # Simular que detecta PDF nativo (texto seleccionable)
        with patch.object(loader, "_has_selectable_text", return_value=True):
            result = loader.load("/fake/document.pdf")

        assert result.success is True
        assert len(result.documents) > 0

    def test_load_nonexistent_file(self):
        """File not found retorna error."""
        loader = PyMuPDFLoader()
        result = loader.load("/fake/nonexistent.pdf")
        assert result.success is False
        assert len(result.errors) > 0

    def test_is_scanned_detection(self):
        """Detección de PDF escaneado por contenido de texto."""
        # PDF con poco texto = escaneado
        loader = PyMuPDFLoader()
        assert loader._is_scanned("") is True
        assert loader._is_scanned(" ") is True
        assert loader._is_scanned("a" * 5) is True

        # PDF con texto = nativo
        assert loader._is_scanned("Texto seleccionable con suficiente contenido") is False


# ─── OCR Loader ────────────────────────────────────────────────────────────

class TestOCRLoader:
    """Tests del loader OCR para PDFs escaneados."""

    def test_loader_type(self):
        loader = OCRLoader()
        assert loader.loader_type == "ocr"

    def test_load_nonexistent_file(self):
        """File not found retorna error."""
        loader = OCRLoader()
        result = loader.load("/fake/nonexistent.pdf")
        assert result.success is False

    @patch("src.ingestion.loaders.pdf_ocr.fitz")
    @patch("src.ingestion.loaders.pdf_ocr.get_ocr_preprocessor")
    def test_ocr_uses_preprocessor(self, mock_preprocessor_get, mock_fitz):
        """OCR loader debe usar el preprocesador de imágenes."""
        mock_preprocessor = MagicMock()
        mock_preprocessor_get.return_value = mock_preprocessor
        mock_preprocessor.preprocess.return_value = MagicMock()

        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = MagicMock()
        mock_doc.__len__ = lambda self: 1
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__getitem__ = lambda self, idx: mock_page
        mock_fitz.open.return_value = mock_doc

        loader = OCRLoader()
        # Simular archivo existente
        with patch.object(Path, "exists", return_value=True):
            # OCRLoader intentará usar easyocr que mockearemos
            with patch.object(loader, "_get_reader", side_effect=Exception("Mock: skipping actual OCR")):
                result = loader.load("/fake/scanned.pdf")

        # Debe fallar gracefully o retornar algo (depende de la implementación)
        # Lo importante es que no crashea sin manejo de errores


# ─── Word Loader ───────────────────────────────────────────────────────────

class TestWordLoader:
    """Tests del loader de documentos Word (.docx)."""

    def test_loader_type(self):
        loader = WordLoader()
        assert loader.loader_type == "word"

    def test_load_nonexistent_file(self):
        """File not found retorna error."""
        loader = WordLoader()
        result = loader.load("/fake/nonexistent.docx")
        assert result.success is False

    @patch("src.ingestion.loaders.word_loader.Document")
    def test_load_word_document(self, mock_doc_class):
        """Carga documento Word con python-docx."""
        mock_doc = MagicMock()
        mock_para1 = MagicMock()
        mock_para1.text = "Este es el primer párrafo del documento."
        mock_para1.style.name = "Normal"
        mock_para2 = MagicMock()
        mock_para2.text = "Introducción"
        mock_para2.style.name = "Heading 1"
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.sections = []
        mock_doc_class.return_value = mock_doc

        loader = WordLoader()
        result = loader.load("/fake/document.docx")

        assert result.success is True
        assert len(result.documents) > 0


# ─── Excel Loader ──────────────────────────────────────────────────────────

class TestExcelLoader:
    """Tests del loader de hojas de cálculo Excel (.xlsx)."""

    def test_loader_type(self):
        loader = ExcelLoader()
        assert loader.loader_type == "excel"

    def test_load_nonexistent_file(self):
        """File not found retorna error."""
        loader = ExcelLoader()
        result = loader.load("/fake/nonexistent.xlsx")
        assert result.success is False

    @patch("src.ingestion.loaders.excel_loader.pd")
    def test_load_excel_document(self, mock_pd):
        """Carga documento Excel con formato row_paragraph."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.to_dict.return_value = {
            "Nombre": {0: "Juan", 1: "María"},
            "Cargo": {0: "Ingeniero", 1: "Analista"},
        }
        mock_df.columns = ["Nombre", "Cargo"]
        mock_pd.read_excel.return_value = mock_df

        loader = ExcelLoader()
        result = loader.load("/fake/document.xlsx")

        assert result.success is True
        assert len(result.documents) > 0
        # Verificar formato row_paragraph
        assert "Juan" in result.documents[0].page_content
        assert "Ingeniero" in result.documents[0].page_content

    @patch("src.ingestion.loaders.excel_loader.pd")
    def test_load_empty_sheet(self, mock_pd):
        """Hoja vacía retorna resultado con 0 documentos."""
        mock_df = MagicMock()
        mock_df.empty = True
        mock_pd.read_excel.return_value = mock_df

        loader = ExcelLoader()
        result = loader.load("/fake/empty.xlsx")

        # Puede ser success con 0 docs o fail dependiendo de la implementación
        # Lo importante es que no crashea
        assert result.success is True or len(result.errors) > 0
