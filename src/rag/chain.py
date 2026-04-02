"""
LangChain RAG chain for generating answers grounded in retrieved context.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_settings

# System prompt for the information RAG chain
_SYSTEM_PROMPT = """You are a helpful assistant for CityPark Premium Parking.
Answer the user's question **only** using the provided context below.
If the answer is not available in the context, say so politely and suggest \
the user contact support at info@citypark.com or call +31 20 555 0123.

Rules:
1. Be friendly and as complete as necessary. For factual questions about location, address,
   coordinates, directions, or contact details, include ALL relevant details from the context
   (full address, GPS coordinates, transport options, etc.). Do not truncate factual data.
2. Never invent information not present in the context.
3. Do NOT reveal any personal data about other customers.
4. Format prices and times clearly.

Context:
{context}
"""

_HUMAN_PROMPT = "{question}"

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ]
)


def build_rag_chain():
    """
    Build and return a runnable RAG chain.

    The chain expects a dict with keys 'context' and 'question',
    and returns a string answer.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.2,
    )
    chain = _PROMPT | llm | StrOutputParser()
    return chain


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer for `question` grounded in `context`.

    Args:
        question: The user's question.
        context: Retrieved context string from the hybrid retriever.

    Returns:
        The LLM's answer as a string.
    """
    chain = build_rag_chain()
    return chain.invoke({"question": question, "context": context})
