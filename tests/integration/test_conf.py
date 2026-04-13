"""
conftest.py para tests de CRAG — auto-mock de dependencias de entorno.

Coloca este archivo en tests/unit/agent/ para que pytest lo cargue
automáticamente en todos los tests de ese directorio.

Propósito:
  Mockear get_llm() a nivel de fixture para que ningún test de la suite
  agent/ realice llamadas reales a AWS Bedrock/ChatBedrock.
  Los tests que necesitan comportamiento específico del LLM pueden
  sobreescribir el mock localmente con @patch en el test mismo.
"""
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_get_llm(monkeypatch):
    """
    Mock global de get_llm para todos los tests de agent/.

    Retorna un MagicMock cuyo .invoke() devuelve un objeto con .content = "".
    Tests que necesitan comportamiento específico sobreescriben con @patch local.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="mock response")
    mock_llm.with_structured_output.return_value = MagicMock(
        invoke=MagicMock(return_value=MagicMock(
            quality="correct", score=0.85, reasoning="mock"
        ))
    )

    monkeypatch.setattr("src.agent.skills.crag.get_llm", lambda **kw: mock_llm)
    monkeypatch.setattr("src.agent.skills.rethinking.get_llm", lambda **kw: mock_llm)
    monkeypatch.setattr("src.agent.skills.query_transformer.get_llm", lambda **kw: mock_llm)
    monkeypatch.setattr("src.agent.skills.answer_validator.get_llm", lambda **kw: mock_llm)
    monkeypatch.setattr("src.config.providers.get_llm", lambda **kw: mock_llm)

    return mock_llm
