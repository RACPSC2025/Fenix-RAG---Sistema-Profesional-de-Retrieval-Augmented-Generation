"""
Text Cleaner — reglas de limpieza de texto pluggables por perfil.

Cada perfil define un conjunto ordenado de CleanerRule que se aplican
al texto extraído de un documento antes del chunking.

Perfiles disponibles:
  "default"        — reglas genéricas para cualquier documento
  "technical"      — documentación técnica (APIs, manuales, guías)
  "ocr_output"     — limpieza post-OCR (artefactos, guiones rotos)
  "contract"       — contratos y acuerdos
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from src.config.logging import get_logger

log = get_logger(__name__)


# ─── Reglas individuales ──────────────────────────────────────────────────────

@dataclass
class CleanerRule:
    """
    Regla atómica de limpieza de texto.

    Aplicada en orden por el TextCleaner para construir
    un pipeline de limpieza composable.
    """
    name: str
    pattern: re.Pattern
    replacement: str = ""
    enabled: bool = True


# ─── Reglas universales (aplican a todos los perfiles) ────────────────────────

def _universal_rules() -> list[CleanerRule]:
    """Reglas que se aplican SIEMPRE, sin importar el perfil."""
    return [
        CleanerRule(
            name="normalize_line_endings",
            pattern=re.compile(r"\r\n?"),
            replacement="\n",
        ),
        CleanerRule(
            name="remove_multiple_blank_lines",
            pattern=re.compile(r"\n{3,}"),
            replacement="\n\n",
        ),
        CleanerRule(
            name="remove_trailing_whitespace",
            pattern=re.compile(r"[ \t]+$", re.MULTILINE),
            replacement="",
        ),
        CleanerRule(
            name="join_broken_words",
            pattern=re.compile(r"(\w)-\n(\w)"),
            replacement=r"\1\2",
        ),
        CleanerRule(
            name="normalize_spaces",
            pattern=re.compile(r" {2,}"),
            replacement=" ",
        ),
        CleanerRule(
            name="remove_control_chars",
            pattern=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
            replacement="",
        ),
    ]


# ─── Reglas por perfil ───────────────────────────────────────────────────────

def _default_rules() -> list[CleanerRule]:
    """Reglas genéricas para cualquier documento."""
    return [
        CleanerRule(
            name="remove_page_numbers",
            pattern=re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE),
            replacement="",
        ),
        CleanerRule(
            name="remove_repeated_headers",
            pattern=re.compile(r"^(.{10,}?)\n\1$", re.MULTILINE),
            replacement=r"\1",
        ),
    ]


def _technical_rules() -> list[CleanerRule]:
    """Reglas para documentación técnica."""
    return [
        CleanerRule(
            name="remove_page_numbers",
            pattern=re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE),
            replacement="",
        ),
        CleanerRule(
            name="normalize_code_blocks",
            pattern=re.compile(r"\n```[a-z]*\n"),
            replacement="\n```",
        ),
        CleanerRule(
            name="normalize_markline_spacing",
            pattern=re.compile(r"\n(#{1,6})\s+", re.MULTILINE),
            replacement=r"\n\1 ",
        ),
    ]


def _ocr_rules() -> list[CleanerRule]:
    """Reglas post-OCR — limpia artefactos de reconocimiento."""
    return [
        CleanerRule(
            name="remove_ocr_artifacts",
            pattern=re.compile(r"[|¡£¢§¦¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿]"),
            replacement="",
        ),
        CleanerRule(
            name="fix_ocr_confused_chars",
            pattern=re.compile(r"(?<!\w)l(?!\w)"),
            replacement="1",
        ),
        CleanerRule(
            name="remove_page_numbers",
            pattern=re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE),
            replacement="",
        ),
        CleanerRule(
            name="remove_repeated_chars",
            pattern=re.compile(r"([a-zA-Z])\1{4,}"),
            replacement=r"\1",
        ),
        CleanerRule(
            name="normalize_broken_lines",
            pattern=re.compile(r"\n\s*([a-z])"),
            replacement=r" \1",
        ),
    ]


def _contract_rules() -> list[CleanerRule]:
    """Reglas para contratos y acuerdos."""
    return [
        CleanerRule(
            name="remove_page_numbers",
            pattern=re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE),
            replacement="",
        ),
        CleanerRule(
            name="remove_repeated_headers",
            pattern=re.compile(r"^(.{10,}?)\n\1$", re.MULTILINE),
            replacement=r"\1",
        ),
        CleanerRule(
            name="normalize_clause_spacing",
            pattern=re.compile(r"(CL[AÁ]USULA|PAR[AÁ]GRAFO|ANEXO)\s*([\d\w]+)", re.IGNORECASE),
            replacement=r"\1 \2",
        ),
    ]


# ─── TextCleaner ──────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Pipeline de limpieza de texto con reglas composables.

    Uso:
        cleaner = TextCleaner(profile="default")
        clean_text = cleaner.clean(raw_text)
    """

    PROFILE_RULES: dict[str, Callable[[], list[CleanerRule]]] = {
        "default": _default_rules,
        "technical": _technical_rules,
        "ocr_output": _ocr_rules,
        "contract": _contract_rules,
    }

    def __init__(self, profile: str = "default") -> None:
        self.profile = profile
        self._rules = self._build_rules(profile)

    @staticmethod
    def _build_rules(profile: str) -> list[CleanerRule]:
        """Construye la lista ordenada de reglas para un perfil."""
        rules = _universal_rules()  # siempre primero
        profile_fn = TextCleaner.PROFILE_RULES.get(profile)
        if profile_fn:
            rules.extend(profile_fn())
        return rules

    def clean(self, text: str) -> str:
        """Aplica todas las reglas al texto."""
        result = text
        for rule in self._rules:
            if rule.enabled:
                result = rule.pattern.sub(rule.replacement, result)
        return result.strip()


# ─── Factory ──────────────────────────────────────────────────────────────────

_cleaner_cache: dict[str, TextCleaner] = {}


def get_cleaner(profile: str = "default") -> TextCleaner:
    """Retorna un TextCleaner cacheado por perfil."""
    if profile not in _cleaner_cache:
        _cleaner_cache[profile] = TextCleaner(profile=profile)
    return _cleaner_cache[profile]
