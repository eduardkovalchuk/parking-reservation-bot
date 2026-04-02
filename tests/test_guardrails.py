"""
Tests for src/guardrails/filters.py

All PII detection is handled by Presidio (spaCy backend).
"""
import pytest

from src.guardrails.filters import GuardrailFilter


@pytest.fixture()
def guardrail():
    return GuardrailFilter()


# ---------------------------------------------------------------------------
# Input guardrail tests
# ---------------------------------------------------------------------------

class TestInputGuardrailClean:
    def test_plain_parking_question_passes(self, guardrail):
        result = guardrail.check_input("What is the hourly parking rate?")
        assert result.blocked is False

    def test_greeting_passes(self, guardrail):
        result = guardrail.check_input("Hello! I want to book a space.")
        assert result.blocked is False

    def test_empty_input_passes(self, guardrail):
        result = guardrail.check_input("")
        assert result.blocked is False

    def test_name_and_car_plate_passes(self, guardrail):
        """Name and plate are legitimate reservation data and must not be blocked."""
        result = guardrail.check_input("My name is John Doe and my plate is ABC-1234")
        assert result.blocked is False


class TestInputGuardrailBlocked:
    def test_credit_card_number_is_blocked(self, guardrail):
        result = guardrail.check_input("My card is 4111111111111111")
        assert result.blocked is True

    def test_iban_is_blocked(self, guardrail):
        result = guardrail.check_input("Transfer to IBAN GB29NWBK60161331926819")
        assert result.blocked is True

    def test_ssn_is_blocked(self, guardrail):
        result = guardrail.check_input("My SSN is 321-12-8393")
        assert result.blocked is True


# ---------------------------------------------------------------------------
# Output guardrail tests
# ---------------------------------------------------------------------------

class TestOutputGuardrail:
    def test_normal_response_passes(self, guardrail):
        result = guardrail.check_output(
            "CityPark is located at Orlyplein 10, Amsterdam. Contact us at info@citypark.com."
        )
        assert result.blocked is False

    def test_response_with_credit_card_is_blocked(self, guardrail):
        result = guardrail.check_output(
            "The customer's card number is 4111111111111111."
        )
        assert result.blocked is True

    def test_own_phone_number_is_whitelisted(self, guardrail):
        result = guardrail.check_output(
            "For assistance call us at +1-555-0123."
        )
        assert result.blocked is False

    def test_own_email_is_whitelisted(self, guardrail):
        result = guardrail.check_output(
            "Contact us at info@citypark.com for more details."
        )
        assert result.blocked is False

    def test_empty_output_passes(self, guardrail):
        result = guardrail.check_output("")
        assert result.blocked is False


# ---------------------------------------------------------------------------
# Anonymisation tests
# ---------------------------------------------------------------------------

class TestAnonymize:
    def test_credit_card_is_anonymised(self, guardrail):
        text = "My Visa number is 4111111111111111."
        assert "4111111111111111" not in guardrail.anonymize(text)

    def test_email_is_anonymised(self, guardrail):
        text = "I've found someones IBAN: GB29NWBK60161331926819"
        assert "GB29NWBK60161331926819" not in guardrail.anonymize(text)

    def test_clean_text_is_not_corrupted(self, guardrail):
        text = "The parking facility welcomes all visitors."
        assert "parking" in guardrail.anonymize(text).lower()

    def test_empty_string_returns_empty(self, guardrail):
        assert guardrail.anonymize("") == ""
