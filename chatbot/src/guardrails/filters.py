"""
Guard-rails for PII detection and sensitive-data filtering.

Uses Microsoft Presidio (spaCy backend) for NLP-based PII entity recognition.

Usage:
    filter_ = GuardrailFilter()

    result = filter_.check_input("My card is 4111 1111 1111 1111")
    if result.blocked:
        print("Blocked:", result.reason)

    clean = filter_.anonymize("Call me at +1-555-0100")
    # → "Call me at <REDACTED>"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presidio (module-level singletons — initialised once at import time)
# ---------------------------------------------------------------------------

# Suppress warnings for spaCy entity labels that Presidio doesn't map to any
# of its own entity types (FAC = facility, NORP = nationalities, etc.).
_nlp_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    "ner_model_configuration": {
        "labels_to_ignore": [
                "FAC",         # facilities / buildings
                "LANGUAGE",    # language names
                "PRODUCT",     # product names (Visa, Apple Pay, CCS Type 2…)
                "QUANTITY",    # measurements (2.1 metres, 50 kW…)
                "CARDINAL",    # plain numbers (150, 24/7…)
                "ORDINAL",     # ordinal numbers (first, second…)
                "PERCENT",     # percentage values
                "MONEY",       # monetary amounts (€15.00, EUR 3.00…)
                "EVENT",       # temporal expressions (today, tomorrow, holidays…)
        ],
    },
}).create_engine()

_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine)
_anonymizer = AnonymizerEngine()

# Entity types that must never appear in chatbot output
_SENSITIVE_ENTITIES = {"CREDIT_CARD", "IBAN_CODE", "US_SSN", "US_BANK_NUMBER", "CRYPTO"}

# Entity types that must never appear in user input
_BLOCKED_INPUT_ENTITIES = {"CREDIT_CARD", "IBAN_CODE", "US_SSN"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Outcome of a guardrail check."""

    blocked: bool = False
    reason: str = ""
    detected_entities: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class GuardrailFilter:
    """Presidio-based guard-rail filter for PII detection and output sanitisation."""

    def _detect(self, text: str) -> List[str]:
        """Return entity types found by Presidio with confidence >= 0.6."""
        results = _analyzer.analyze(
            text=text, language="en", entities=list(_SENSITIVE_ENTITIES)
        )
        return [r.entity_type for r in results if r.score >= 0.6]

    def check_input(self, text: str) -> GuardrailResult:
        """
        Block input that contains sensitive financial/identity PII.
        Name, car plate, phone, and email are allowed (needed for reservations).
        """
        if not text or not text.strip():
            return GuardrailResult()

        entities = self._detect(text)
        blocked = [e for e in entities if e in _BLOCKED_INPUT_ENTITIES]
        if blocked:
            entity_label = blocked[0].replace("_", " ").replace("US ", "")
            return GuardrailResult(
                blocked=True,
                reason=f"For your security, please do not share {entity_label} information in chat.",
                detected_entities=blocked,
            )

        return GuardrailResult()

    def check_output(self, text: str) -> GuardrailResult:
        """
        Block responses that leak sensitive PII.
        Own contact details (info@citypark.com, +31 20 555 0123) are whitelisted.
        """
        if not text or not text.strip():
            return GuardrailResult()

        # Strip whitelisted contact details before analysis
        scrubbed = text.replace("info@citypark.com", "").replace("+31 20 555 0123", "")

        entities = self._detect(scrubbed)
        sensitive = [e for e in entities if e in _SENSITIVE_ENTITIES]
        if sensitive:
            return GuardrailResult(
                blocked=True,
                reason="Output contains potentially sensitive financial data.",
                detected_entities=sensitive,
            )

        return GuardrailResult(detected_entities=entities)

    def anonymize(self, text: str) -> str:
        """Replace detected PII in `text` with placeholder tokens using Presidio."""
        if not text:
            return text

        analysis = _analyzer.analyze(
            text=text, language="en", entities=list(_SENSITIVE_ENTITIES)
        )
        if not analysis:
            return text

        return _anonymizer.anonymize(
            text=text,
            analyzer_results=analysis,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})},
        ).text
