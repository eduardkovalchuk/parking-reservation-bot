"""
CityPark Parking Reservation Chatbot — Streamlit UI
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
import streamlit as st
import weaviate as weaviate_lib

from src.chatbot.graph import build_graph, chat
from src.config import get_settings
from src.database.vector_store import get_weaviate_client

# ---------------------------------------------------------------------------
# Service health check
# ---------------------------------------------------------------------------

def _check_services() -> dict[str, bool]:
    """Probe Weaviate and PostgreSQL. Returns connection status for each."""
    settings = get_settings()
    status = {"weaviate": False, "postgres": False}

    try:
        client = weaviate_lib.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_port,
            grpc_port=settings.weaviate_grpc_port,
        )
        status["weaviate"] = client.is_ready()
        client.close()
    except Exception:
        pass

    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=2,
        )
        conn.close()
        status["postgres"] = True
    except Exception:
        pass

    return status


# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """Build the LangGraph app once per browser session."""
    if "app" in st.session_state:
        return

    try:
        client = get_weaviate_client()
        app = build_graph(client)
        st.session_state.weaviate_client = client
        st.session_state.app = app
    except Exception as exc:
        st.error(f"Failed to initialise the chatbot: {exc}")
        st.stop()

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm the CityPark Parking Assistant. I can help you with:\n\n"
                "- **Parking information** — prices, availability, location, hours, policies\n"
                "- **Reservations** — I'll guide you through booking a space step by step\n\n"
                "How can I help you today?"
            ),
        }
    ]
    st.session_state.reservation_data = {}


def _reset_session() -> None:
    """Close connections and clear session so the next rerun starts fresh."""
    if client := st.session_state.get("weaviate_client"):
        try:
            client.close()
        except Exception:
            pass

    for key in ["app", "weaviate_client", "messages", "reservation_data"]:
        st.session_state.pop(key, None)

    st.rerun()


# ---------------------------------------------------------------------------
# Read reservation_data from the graph's MemorySaver
# ---------------------------------------------------------------------------

def _get_reservation_data() -> dict:
    config = {"configurable": {"thread_id": "default"}}
    try:
        snapshot = st.session_state.app.get_state(config)
        return snapshot.values.get("reservation_data") or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(status: dict[str, bool]) -> None:
    with st.sidebar:
        st.title("CityPark Assistant")
        st.divider()

        # ── Service status ────────────────────────────────────────────────
        st.subheader("Services")
        for name, ok in status.items():
            icon = "🟢" if ok else "🔴"
            label = name.capitalize()
            st.markdown(f"{icon} &nbsp; **{label}**", unsafe_allow_html=True)

        st.divider()

        # ── Reservation panel ─────────────────────────────────────────────
        st.subheader("Reservation Details")
        data: dict = st.session_state.get("reservation_data", {})

        if not data:
            st.info("No reservation in progress.")
        else:
            if data.get("name") or data.get("surname"):
                st.markdown(f"**Guest** &nbsp; {data.get('name', '')} {data.get('surname', '')}".strip())
            if data.get("car_number"):
                st.markdown(f"**Plate** &nbsp; {data['car_number']}")
            if data.get("space_type"):
                st.markdown(f"**Space type** &nbsp; {data['space_type'].capitalize()}")
            if data.get("start_datetime"):
                st.markdown(f"**From** &nbsp; {data['start_datetime']}")
            if data.get("end_datetime"):
                st.markdown(f"**To** &nbsp; {data['end_datetime']}")
            if data.get("floor") and data.get("space_number"):
                st.markdown(f"**Location** &nbsp; Floor {data['floor']}, Space {data['space_number']}")
            if data.get("reservation_id"):
                st.markdown(f"**Reservation ID** &nbsp; #{data['reservation_id']}")
            if data.get("total_cost") is not None:
                st.markdown(f"**Total cost** &nbsp; €{float(data['total_cost']):.2f}")

        st.divider()

        # ── New conversation ──────────────────────────────────────────────
        if st.button("Reset conversation", type="secondary", use_container_width=True):
            _reset_session()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="CityPark Parking Assistant",
        page_icon="🚗",
        layout="wide",
    )

    _init_session()
    status = _check_services()
    _render_sidebar(status)

    st.header("🚗 CityPark Premium Parking Assistant")
    st.caption("Ask about prices, availability, location, or make a reservation.")

    # ── Render existing messages ──────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Handle new input ──────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about parking or make a reservation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = chat(st.session_state.app, prompt)
                except Exception as exc:
                    reply = (
                        "I'm sorry, something went wrong on my end. "
                        "Please try again or contact info@citypark.com or +31 20 555 0123."
                    )

            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.reservation_data = _get_reservation_data()
        st.rerun()


main()
