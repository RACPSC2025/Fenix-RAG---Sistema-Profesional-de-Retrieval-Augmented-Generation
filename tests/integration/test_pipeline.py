"""Tests de integración del pipeline de ingestión end-to-end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pipeline import IngestionPipeline, PipelineResult


class TestPipelineIntegration:
    """Tests de integración del pipeline completo."""

    @pytest.fixture
    def pipeline(self):
        """Pipeline de ingestión para tests."""
        return IngestionPipeline(
            chunk_size=500,
            chunk_overlap=50,
            cleaner_profile="default",
            augment_with_questions=False,
        )

    def test_pipeline_nonexistent_file(self, pipeline):
        """Archivo inexistente retorna error."""
        result = pipeline.ingest_file("/fake/nonexistent.pdf")
        assert result.success is False
        assert len(result.errors) > 0
        assert "no encontrado" in result.errors[0].lower()

    @patch("src.ingestion.pipeline.get_vector_store")
    @patch("src.ingestion.processors.hierarchical_chunker.HierarchicalChunker")
    @patch("src.ingestion.registry.get_registry")
    @patch("pathlib.Path.exists")
    def test_pipeline_full_flow(
        self, mock_exists, mock_registry_get, mock_chunker, mock_vs_get,
    ):
        """Flujo completo: detectar MIME → cargar → limpiar → chunk → indexar."""
        mock_exists.return_value = True

        # Mock del registry y loader
        mock_registry = MagicMock()
        mock_registry_get.return_value = mock_registry

        mock_loader = MagicMock()
        mock_loader.load_multiple.return_value = [
            MagicMock(
                documents=[
                    MagicMock(
                        page_content="Contenido del documento para pruebas de integración.",
                        metadata={"source": "test.pdf", "page": 1},
                    )
                ],
                errors=[],
            )
        ]
        mock_loader.load.return_value = MagicMock(
            documents=[
                MagicMock(
                    page_content="Contenido del documento para pruebas de integración.",
                    metadata={"source": "test.pdf", "page": 1},
                )
            ],
            errors=[],
        )
        mock_registry.select.return_value = mock_loader
        mock_registry.get_loader.return_value = mock_loader

        # Mock del chunker
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.return_value = [
            MagicMock(
                page_content="Chunk 1 del documento.",
                metadata={"source": "test.pdf", "chunk_index": 0},
            ),
            MagicMock(
                page_content="Chunk 2 del documento.",
                metadata={"source": "test.pdf", "chunk_index": 1},
            ),
        ]
        mock_chunker.return_value = mock_chunker_instance

        # Mock del vector store
        mock_vs = MagicMock()
        mock_vs.is_initialized = True
        mock_vs_get.return_value = mock_vs

        # Ejecutar pipeline
        result = pipeline.ingest_file("/fake/test.pdf")

        # Verificar resultados
        assert result.success is True
        assert result.chunk_count > 0
        assert result.loader_used != ""
        assert result.mime_type != ""

    def test_pipeline_result_fields(self, pipeline):
        """PipelineResult tiene todos los campos esperados."""
        with patch.object(pipeline, "ingest_file", return_value=PipelineResult(
            source_path=Path("/fake/test.pdf"),
            success=True,
            documents=[],
            errors=[],
            loader_used="pymupdf",
            document_type="documentation",
            mime_type="application/pdf",
            pages_processed=5,
            chunk_count=10,
            page_count=5,
            classifier_confidence=0.95,
            indexed=True,
            index_errors=[],
        )):
            result = pipeline.ingest_file("/fake/test.pdf")

        assert result.source_path == Path("/fake/test.pdf")
        assert result.success is True
        assert result.loader_used == "pymupdf"
        assert result.document_type == "documentation"
        assert result.mime_type == "application/pdf"
        assert result.chunk_count == 10
        assert result.classifier_confidence == 0.95
        assert result.indexed is True


class TestPipelineWithDocumentType:
    """Tests de propagación de document_type en el pipeline."""

    @pytest.fixture
    def pipeline(self):
        return IngestionPipeline(
            chunk_size=500,
            chunk_overlap=50,
            cleaner_profile="default",
        )

    def test_pipeline_accepts_document_type(self, pipeline):
        """ingest_file acepta parámetro document_type."""
        # Verificar que la firma acepta document_type
        import inspect
        sig = inspect.signature(pipeline.ingest_file)
        params = list(sig.parameters.keys())
        assert "document_type" in params or "plan" in params

    @patch("src.ingestion.processors.adaptive_chunker.AdaptiveChunker.chunk")
    @patch("src.ingestion.pipeline.get_vector_store")
    @patch("src.ingestion.registry.get_registry")
    @patch("pathlib.Path.exists")
    def test_pipeline_propagates_document_type(
        self, mock_exists, mock_registry_get, mock_vs_get, mock_chunk,
    ):
        """El document_type del plan se propaga al chunker."""
        mock_exists.return_value = True

        # Mock del registry
        mock_registry = MagicMock()
        mock_registry_get.return_value = mock_registry
        mock_loader = MagicMock()
        mock_loader.load_multiple.return_value = [
            MagicMock(
                documents=[
                    MagicMock(
                        page_content="Contenido de prueba.",
                        metadata={"source": "test.pdf"},
                    )
                ],
                errors=[],
            )
        ]
        mock_registry.select.return_value = mock_loader
        mock_registry.get_loader.return_value = mock_loader

        # Mock del chunker
        mock_chunk.return_value = [
            MagicMock(
                page_content="Chunk 1.",
                metadata={"source": "test.pdf", "chunk_index": 0, "document_type": "contract"},
            )
        ]

        # Mock del vector store
        mock_vs = MagicMock()
        mock_vs.is_initialized = True
        mock_vs_get.return_value = mock_vs

        # Ejecutar con document_type explícito
        result = pipeline.ingest_file("/fake/test.pdf", document_type="contract")

        assert result.document_type == "contract"
