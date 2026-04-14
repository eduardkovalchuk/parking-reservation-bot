"""
LangGraph node functions for the parking reservation chatbot.

Node execution order (happy path):
  input_guardrail → classify_intent → retrieve_context → generate_response
                                    → handle_reservation
  All terminal nodes → output_guardrail → END
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict

import httpx
import weaviate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.chatbot.state import ChatState
from src.config import get_settings
from src.guardrails.filters import GuardrailFilter
from src.rag import chain as rag_chain
from src.rag import retriever as rag_retriever

logger = logging.getLogger(__name__)


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.openai_chat_model, openai_api_key=s.openai_api_key, temperature=0.1)


def _parse_llm_json(text: str) -> dict:
    """Strip optional markdown code fences and parse JSON from LLM output."""
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner = [line for line in lines[1:] if line.strip() != "```"]
        content = "\n".join(inner).strip()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Guardrail nodes
# ---------------------------------------------------------------------------

def node_input_guardrail(state: ChatState) -> Dict[str, Any]:
    """Block input that contains sensitive financial/identity PII."""
    guardrail = GuardrailFilter()
    user_input = state.get("user_input", "")

    result = guardrail.check_input(user_input)
    if result.blocked:
        logger.info("Input blocked by guardrail: %s", result.reason)
        return {
            "input_blocked": True,
            "block_reason": result.reason,
        }

    # Anonymise any detected PII in the stored user input
    cleaned = guardrail.anonymize(user_input)
    return {
        "input_blocked": False,
        "block_reason": "",
        "user_input": cleaned,
    }


def node_output_guardrail(state: ChatState) -> Dict[str, Any]:
    """
    Inspect the last AI message to ensure no sensitive data is leaked before
    it reaches the user.
    """
    guardrail = GuardrailFilter()
    messages = state.get("messages", [])
    if not messages:
        return {"output_blocked": False}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return {"output_blocked": False}

    result = guardrail.check_output(last_msg.content)
    if result.blocked:
        logger.warning("Output blocked by guardrail: %s", result.reason)
        safe_reply = (
            "I'm sorry, I can't provide that information for privacy reasons. "
            "Please contact us directly at info@citypark.com or +31 20 555 0123."
        )
        return {
            "output_blocked": True,
            "messages": [AIMessage(content=safe_reply)],
        }

    # Replace with anonymised version if needed
    anonymised = guardrail.anonymize(last_msg.content)
    if anonymised != last_msg.content:
        return {"messages": [AIMessage(content=anonymised)]}

    return {"output_blocked": False}


def node_blocked_response(state: ChatState) -> Dict[str, Any]:
    """Return a polite refusal when input has been blocked by guardrails."""
    reason = state.get("block_reason", "Your message could not be processed.")
    reply = (
        f"I'm sorry, I cannot process that request. {reason} "
        "If you have questions about our parking facility, I'm happy to help!"
    )
    return {"messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_INTENT_PROMPT = """You are an intent classifier for a parking reservation chatbot.

Classify the user's message into exactly ONE of these intents:
- information   : User wants information about the parking (prices, location, hours, amenities,
                  policies, booking process, how to reserve, etc.). This includes questions like
                  "how do I book?" or "what is the reservation process?" — the user is asking
                  for information, NOT actively trying to make a booking.
- reservation   : User is ACTIVELY requesting to make, modify, or cancel a specific reservation
                  right now (e.g. "I want to book a space", "reserve a spot for me", "cancel my booking").
                  Do NOT use this for questions about how reservations work.
- general       : General small-talk or greetings that you can answer without parking info
- off_topic     : User is asking about something completely unrelated to parking

Consider conversation history when classifying.

User message: {message}

Respond with a single JSON object: {{"intent": "<intent>"}}
"""


def node_classify_intent(state: ChatState) -> Dict[str, Any]:
    """Use the LLM to classify the user's intent."""
    llm = _get_llm()
    user_input = state.get("user_input", "")

    # If we are mid-reservation, stay in reservation flow
    reservation_stage = state.get("reservation_stage", "idle")
    if reservation_stage not in ("idle", "completed", "cancelled"):
        return {"intent": "reservation"}

    prompt = _INTENT_PROMPT.format(message=user_input)
    try:
        response = llm.invoke(prompt)
        parsed = _parse_llm_json(response.content)
        intent = parsed.get("intent", "general")
    except Exception as exc:
        logger.warning("Intent classification failed (%s); defaulting to 'general'.", exc)
        intent = "general"

    return {"intent": intent}


# ---------------------------------------------------------------------------
# Information flow
# ---------------------------------------------------------------------------

def node_retrieve_context(state: ChatState, weaviate_client: weaviate.WeaviateClient) -> Dict[str, Any]:
    """Retrieve relevant context from Weaviate and/or PostgreSQL."""
    query = state.get("user_input", "")
    settings = get_settings()
    result = rag_retriever.retrieve(query, weaviate_client, k=settings.retrieval_k)
    return {"retrieved_context": result.combined_context}


def node_generate_response(state: ChatState) -> Dict[str, Any]:
    """Generate an answer using the retrieved context (RAG chain)."""
    question = state.get("user_input", "")
    context = state.get("retrieved_context", "")

    if not context:
        answer = (
            "I don't have specific information on that right now. "
            "Please contact us at info@citypark.com or +31 20 555 0123 and we'll be happy to help."
        )
    else:
        answer = rag_chain.generate_answer(question, context)

    return {"messages": [AIMessage(content=answer)]}


def node_general_response(state: ChatState) -> Dict[str, Any]:
    """Handle general small-talk and greetings without RAG retrieval."""
    llm = _get_llm()
    user_input = state.get("user_input", "")
    system = (
        "You are a friendly assistant for CityPark Premium Parking, a single parking facility "
        "located at Orlyplein 10, 1043 DP Amsterdam (Sloterdijk). "
        "There is only ONE location — do NOT mention or imply multiple locations or cities. "
        "Respond to the user's greeting or general question briefly and invite them "
        "to ask about parking information or make a reservation."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
    reply = llm.invoke(messages)
    return {"messages": [AIMessage(content=reply.content)]}


def node_off_topic_response(state: ChatState) -> Dict[str, Any]:
    """Politely redirect off-topic requests back to parking topics."""
    reply = (
        "I'm specialised in CityPark Parking assistance and can only help with "
        "parking-related questions and reservations. Is there anything about "
        "our parking facility I can help you with?"
    )
    return {"messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# Reservation flow
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """Extract the value for the field '{field}' from the conversation below.
Return a JSON object with key '{field}'. If the value is not present, return {{'{field}': null}}.

Field descriptions:
- name: The customer's first name
- surname: The customer's last name / family name
- car_number: The vehicle licence/registration plate number (e.g. "ABC-1234")
- start_datetime: The reservation start date and time (ISO 8601, e.g. "2026-04-01T09:00:00")
- end_datetime: The reservation end date and time (ISO 8601)

Today's date for reference: {today}

Already collected fields (use these to resolve relative expressions):
{collected}

Conversation (most recent last):
{history}

Latest user message: {message}

Important: Return ONLY the JSON object, no extra text.
"""

_CONFIRMATION_WORDS = {"yes", "y", "confirm", "ok", "okay", "sure", "correct", "right", "yep", "yeah"}
_CANCELLATION_WORDS = {"no", "n", "cancel", "stop", "abort", "quit", "exit", "nevermind", "nope"}


def _extract_field(
    field: str,
    message: str,
    llm: ChatOpenAI,
    history: list | None = None,
    collected: dict | None = None,
) -> Any:
    """Use the LLM to extract a specific field value from the user message.

    Pass the last few conversation messages as `history` so the LLM can
    understand short replies (e.g. "Eduard") in context of the previous
    bot question (e.g. "What is your first name?").

    Pass already-collected `collected` fields so relative expressions like
    "until 6 pm" are resolved against the known start_datetime.
    """
    history_text = ""
    if history:
        lines = []
        for msg in history:
            role = "Assistant" if isinstance(msg, AIMessage) else "User"
            lines.append(f"{role}: {msg.content}")
        history_text = "\n".join(lines)

    collected_text = "(none yet)"
    if collected:
        collected_text = "\n".join(
            f"- {k}: {v}" for k, v in collected.items() if not k.startswith("_")
        ) or "(none yet)"

    prompt = _EXTRACT_PROMPT.format(
        field=field,
        message=message,
        history=history_text or "(no prior context)",
        collected=collected_text,
        today=datetime.now().strftime("%Y-%m-%d"),
    )
    try:
        response = llm.invoke(prompt)
        parsed = _parse_llm_json(response.content)
        return parsed.get(field)
    except Exception as exc:
        logger.warning("Field extraction failed for '%s': %s", field, exc)
        return None


_FIELD_QUESTIONS = {
    "need_name": "What is your **first name**?",
    "need_surname": "What is your **last name** (surname)?",
    "need_car_number": "What is your **vehicle registration plate number** (e.g. ABC-1234)?",
    "need_start": (
        "What **date and time** would you like your reservation to **start**? "
        "(e.g. '2026-04-01 09:00' or 'April 1st at 9am')"
    ),
    "need_end": (
        "What **date and time** would you like your reservation to **end**? "
        "(e.g. '2026-04-01 17:00' or 'April 1st at 5pm')"
    ),
}

_STAGE_TO_FIELD = {
    "need_name": "name",
    "need_surname": "surname",
    "need_car_number": "car_number",
    "need_start": "start_datetime",
    "need_end": "end_datetime",
}

_STAGE_SEQUENCE = [
    "need_name",
    "need_surname",
    "need_car_number",
    "need_start",
    "need_end",
    "confirming",
]

# Quick regex for unmistakable abort words (avoids an LLM call in obvious cases)
_ABORT_RE = re.compile(
    r"^(stop|cancel|abort|quit|exit|never\s*mind|forget\s*it|leave\s*it|no\s*thanks|i\s*don'?t\s*want)\b",
    re.IGNORECASE,
)

_ABORT_CHECK_PROMPT = """The user is currently filling in a parking reservation form.
Is the user trying to STOP, CANCEL or ABANDON the reservation process?

Answer true only for clear abandonment signals like:
  "stop", "cancel", "I want to stop", "forget it", "never mind", "I changed my mind", "abort"

Answer false for anything that looks like a form answer:
  a name, plate number, date, time, "yes", "no", or a short factual reply.

User message: {message}

Respond with ONLY a JSON object: {{"abort": true}} or {{"abort": false}}
"""


def _is_abort_intent(message: str, llm: ChatOpenAI) -> bool:
    """Return True if the user is trying to abort the reservation flow."""
    if _ABORT_RE.search(message.strip()):
        return True
    try:
        prompt = _ABORT_CHECK_PROMPT.format(message=message)
        response = llm.invoke(prompt)
        parsed = _parse_llm_json(response.content)
        return bool(parsed.get("abort", False))
    except Exception:
        return False


def _format_confirmation_summary(data: dict) -> str:
    summary = (
        "Please confirm your reservation details:\n\n"
        f"  • **Name**: {data.get('name', '–')} {data.get('surname', '–')}\n"
        f"  • **Car number**: {data.get('car_number', '–')}\n"
        f"  • **Start**: {data.get('start_datetime', '–')}\n"
        f"  • **End**: {data.get('end_datetime', '–')}\n\n"
        "Type **yes** to confirm or **no** to cancel."
    )
    return summary


def node_handle_reservation(state: ChatState) -> Dict[str, Any]:
    """
    Multi-turn reservation state machine node.

    This single node manages the full reservation collection flow by inspecting
    and advancing the 'reservation_stage' in the state.
    """
    llm = _get_llm()
    user_input = state.get("user_input", "")
    stage = state.get("reservation_stage", "idle")
    res_data: dict = dict(state.get("reservation_data") or {})

    # ── Abort confirmation ─────────────────────────────────────────────────
    if stage == "abort_confirming":
        normalized = user_input.strip().lower()
        resumed_stage = res_data.pop("_pre_abort_stage", "idle")
        if normalized in _CONFIRMATION_WORDS:
            return {
                "reservation_stage": "idle",
                "reservation_data": {},
                "messages": [
                    AIMessage(
                        content=(
                            "Your reservation has been cancelled. "
                            "Feel free to ask me anything else or start a new booking whenever you’re ready!"
                        )
                    )
                ],
            }
        # User said no — resume where we left off
        if resumed_stage in _FIELD_QUESTIONS:
            reply = "No problem! Let’s continue.\n\n" + _FIELD_QUESTIONS[resumed_stage]
        elif resumed_stage == "confirming":
            reply = "No problem! Let’s continue.\n\n" + _format_confirmation_summary(res_data)
        else:
            reply = "No problem! Let’s continue."
        return {
            "reservation_stage": resumed_stage,
            "reservation_data": res_data,
            "messages": [AIMessage(content=reply)],
        }

    # ── Start new reservation (handled below after completed/cancelled reset) ──

    # ── Collecting fields ──────────────────────────────────────────────────
    if stage in _STAGE_TO_FIELD:
        # Check if the user is trying to abandon the form before extracting.
        if _is_abort_intent(user_input, llm):
            res_data["_pre_abort_stage"] = stage
            return {
                "reservation_stage": "abort_confirming",
                "reservation_data": res_data,
                "messages": [
                    AIMessage(
                        content=(
                            "Are you sure you want to cancel the booking process? "
                            "Type **yes** to stop, or **no** to continue where you left off."
                        )
                    )
                ],
            }

        field_name = _STAGE_TO_FIELD[stage]
        # Pass the last 4 messages so the LLM sees the bot's question alongside
        # the user's (potentially short) answer.
        recent_history = state.get("messages", [])[-4:]
        extracted = _extract_field(
            field_name, user_input, llm,
            history=recent_history,
            collected=res_data,
        )

        if extracted is None:
            reply = (
                f"I couldn't extract your {field_name.replace('_', ' ')}. "
                f"Could you please try again? {_FIELD_QUESTIONS[stage]}"
            )
            return {
                "reservation_stage": stage,
                "reservation_data": res_data,
                "messages": [AIMessage(content=reply)],
            }

        res_data[field_name] = str(extracted)
        current_idx = _STAGE_SEQUENCE.index(stage)
        next_stage = _STAGE_SEQUENCE[current_idx + 1]

        if next_stage == "confirming":
            reply = _format_confirmation_summary(res_data)
        else:
            reply = _FIELD_QUESTIONS[next_stage]

        return {
            "reservation_stage": next_stage,
            "reservation_data": res_data,
            "messages": [AIMessage(content=reply)],
        }

    # ── Confirmation ───────────────────────────────────────────────────────
    if stage == "confirming":
        normalized = user_input.strip().lower()

        if normalized in _CANCELLATION_WORDS:
            return {
                "reservation_stage": "cancelled",
                "reservation_data": {},
                "messages": [
                    AIMessage(
                        content=(
                            "Your reservation has been cancelled. "
                            "Feel free to start a new booking whenever you're ready!"
                        )
                    )
                ],
            }

        if normalized not in _CONFIRMATION_WORDS:
            return {
                "reservation_stage": "confirming",
                "reservation_data": res_data,
                "messages": [
                    AIMessage(
                        content=(
                            _format_confirmation_summary(res_data)
                            + "\n\nPlease type **yes** to confirm or **no** to cancel."
                        )
                    )
                ],
            }

        # Save the reservation via Admin API
        try:
            settings = get_settings()
            payload = {
                "customer_name": res_data["name"],
                "customer_surname": res_data["surname"],
                "car_number": res_data["car_number"],
                "start_datetime": res_data["start_datetime"],
                "end_datetime": res_data["end_datetime"],
                "space_type": res_data.get("space_type", "standard"),
            }
            resp = httpx.post(
                f"http://{settings.admin_api_host}:{settings.admin_api_port}/api/reservations",
                json=payload,
                timeout=10.0,
            )

            if resp.status_code == 409:
                return {
                    "reservation_stage": "idle",
                    "reservation_data": {},
                    "messages": [
                        AIMessage(
                            content=(
                                "I'm sorry, there are no available spaces of the requested type right now. "
                                "Please try again later or contact us at info@citypark.com or +31 20 555 0123."
                            )
                        )
                    ],
                }

            resp.raise_for_status()
            reservation = resp.json()

            res_data["reservation_id"] = reservation["id"]
            res_data["total_cost"] = float(reservation["total_cost"])
            res_data["space_id"] = reservation["space_id"]

            start_dt = datetime.fromisoformat(res_data["start_datetime"])
            end_dt = datetime.fromisoformat(res_data["end_datetime"])

            reply = (
                f"✅ **Reservation submitted!**\n\n"
                f"  • Reservation ID: **#{reservation['id']}**\n"
                f"  • Space: Floor {reservation['floor']}, Space {reservation['space_number']}\n"
                f"  • Name: {res_data['name']} {res_data['surname']}\n"
                f"  • Car: {res_data['car_number']}\n"
                f"  • Start: {start_dt.strftime('%Y-%m-%d %H:%M')}\n"
                f"  • End: {end_dt.strftime('%Y-%m-%d %H:%M')}\n"
                f"  • Estimated cost: **€{float(reservation['total_cost']):.2f}**\n\n"
                "Your reservation is pending administrator approval. "
                "You will be notified once it is approved. Thank you for choosing CityPark!"
            )
        except Exception as exc:
            logger.error("Failed to save reservation: %s", exc)
            reply = (
                "I'm sorry, there was an issue saving your reservation. "
                "Please try again or contact us directly at info@citypark.com or +31 20 555 0123."
            )
            return {
                "reservation_stage": "idle",
                "reservation_data": {},
                "messages": [AIMessage(content=reply)],
            }

        return {
            "reservation_stage": "completed",
            "reservation_data": res_data,
            "messages": [AIMessage(content=reply)],
        }

    # ── Post-completion / already done ────────────────────────────────────
    # Reset so the next block starts a fresh booking flow
    if stage in ("completed", "cancelled"):
        stage = "idle"
        res_data = {}

    # ── Start new reservation ──────────────────────────────────────────────
    if stage == "idle":
        reply = (
            "I'd be happy to help you book a parking space! "
            "Let me collect a few details.\n\n" + _FIELD_QUESTIONS["need_name"]
        )
        return {
            "reservation_stage": "need_name",
            "reservation_data": res_data,
            "messages": [AIMessage(content=reply)],
        }

    return {}
