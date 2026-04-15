"""
CityPark Parking Reservation Chatbot — Streamlit UI
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import psycopg
import psycopg2
import streamlit as st
import weaviate as weaviate_lib
from langgraph.checkpoint.postgres import PostgresSaver

from src.chatbot.graph import (
    ChatResult,
    build_graph,
    chat,
    resume_after_admin_decision,
)
from src.config import get_settings
from src.database import sql_store
from src.database.vector_store import get_weaviate_client

# ---------------------------------------------------------------------------
# Service health check
# ---------------------------------------------------------------------------

def _check_services() -> dict[str, bool]:
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

    settings = get_settings()

    try:
        weaviate_client = get_weaviate_client()
    except Exception as exc:
        st.error(f"Failed to connect to Weaviate: {exc}")
        st.stop()

    try:
        conn_string = (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        postgres_conn = psycopg.connect(conn_string, autocommit=True)
        checkpointer = PostgresSaver(postgres_conn)
        checkpointer.setup()
        st.session_state.postgres_conn = postgres_conn
    except Exception as exc:
        st.error(f"Failed to set up Postgres checkpointer: {exc}")
        weaviate_client.close()
        st.stop()

    try:
        app = build_graph(weaviate_client, checkpointer)
        st.session_state.weaviate_client = weaviate_client
        st.session_state.app = app
    except Exception as exc:
        st.error(f"Failed to initialise the chatbot: {exc}")
        weaviate_client.close()
        postgres_conn.close()
        st.stop()

    st.session_state.thread_id = str(uuid.uuid4())
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
    st.session_state.awaiting_approval_id = None


def _reset_session() -> None:
    """Close connections and clear session so the next rerun starts fresh."""
    if client := st.session_state.get("weaviate_client"):
        try:
            client.close()
        except Exception:
            pass

    if conn := st.session_state.get("postgres_conn"):
        try:
            conn.close()
        except Exception:
            pass

    for key in ["app", "weaviate_client", "postgres_conn", "messages",
                "reservation_data", "awaiting_approval_id"]:
        st.session_state.pop(key, None)

    st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_reservation_data() -> dict:
    config = {"configurable": {"thread_id": st.session_state.get("thread_id", "default")}}
    try:
        snapshot = st.session_state.app.get_state(config)
        return snapshot.values.get("reservation_data") or {}
    except Exception:
        return {}


def _get_reservation_status(reservation_id: int) -> Optional[str]:
    """Poll DB for the current status of a reservation."""
    try:
        row = sql_store.get_reservation_by_id(reservation_id)
        return row["status"] if row else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Approval polling
# ---------------------------------------------------------------------------

def _check_for_admin_decision() -> None:
    """
    Called on each rerun while awaiting_approval_id is set.
    Polls the DB; when the status changes from 'pending', resumes the graph.
    """
    reservation_id: int = st.session_state.awaiting_approval_id
    status = _get_reservation_status(reservation_id)

    if status and status != "pending":
        with st.spinner("Admin decided — processing your reservation…"):
            result = resume_after_admin_decision(
                st.session_state.app, status, reservation_id,
                thread_id=st.session_state.thread_id,
            )
        st.session_state.messages.append({"role": "assistant", "content": result.reply})
        st.session_state.reservation_data = _get_reservation_data()
        st.session_state.awaiting_approval_id = None
        st.rerun()
    else:
        # Still pending — wait and poll again
        time.sleep(4)
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(status: dict[str, bool]) -> None:
    with st.sidebar:
        st.title("CityPark Assistant")
        st.divider()

        st.subheader("Services")
        for name, ok in status.items():
            icon = "🟢" if ok else "🔴"
            st.markdown(f"{icon} &nbsp; **{name.capitalize()}**", unsafe_allow_html=True)

        st.divider()

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

            # Show approval status badge
            if data.get("reservation_id") and st.session_state.get("awaiting_approval_id"):
                st.warning("⏳ Pending admin approval")

        st.divider()

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

    # Polling loop — runs on every rerun while waiting for admin
    if st.session_state.get("awaiting_approval_id"):
        _check_for_admin_decision()
        # _check_for_admin_decision calls st.rerun() or st.stop() so we
        # only reach here if it returned normally (shouldn't happen normally)

    status = _check_services()
    _render_sidebar(status)

    st.header("🚗 CityPark Premium Parking Assistant")
    st.caption("Ask about prices, availability, location, or make a reservation.")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show waiting banner if interrupted
    if st.session_state.get("awaiting_approval_id"):
        rid = st.session_state.awaiting_approval_id
        with st.chat_message("assistant"):
            st.info(f"⏳ Reservation #{rid} is awaiting admin approval. This page refreshes automatically…")
        return  # Don't accept new input while paused

    # Normal chat input
    if prompt := st.chat_input("Ask about parking or make a reservation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result: ChatResult = chat(
                        st.session_state.app, prompt,
                        thread_id=st.session_state.thread_id,
                    )
                except Exception as exc:
                    result = ChatResult(
                        reply=(
                            "I'm sorry, something went wrong on my end. "
                            "Please try again or contact info@citypark.com."
                        )
                    )

            st.markdown(result.reply)

        st.session_state.messages.append({"role": "assistant", "content": result.reply})
        st.session_state.reservation_data = _get_reservation_data()

        if result.interrupted and result.interrupt_reservation_id:
            st.session_state.awaiting_approval_id = result.interrupt_reservation_id

        st.rerun()


main()
