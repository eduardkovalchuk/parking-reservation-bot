"""
CityPark MCP Server — writes confirmed reservations to a log file.

Exposed tool:
    write_confirmed_reservation — validates input and appends one line to the log.

Security:
    Every call must supply the correct api_key (shared secret).  The output
    path is fixed server-side; callers cannot choose or traverse paths.

Transport:  streamable-http  (POST /mcp/)
Run:        python server.py
"""
from __future__ import annotations

import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("CityPark Reservation Logger")

# ---------------------------------------------------------------------------
# Configuration — read once at startup
# ---------------------------------------------------------------------------

_API_KEY: str = os.environ.get("MCP_API_KEY", "")
_RESERVATIONS_FILE = Path(
    os.environ.get(
        "RESERVATIONS_FILE",
        "/data/reservations/confirmed_reservations.txt",
    )
)

# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

_CAR_NUMBER_RE = re.compile(r"^[A-Z0-9\-]{2,12}$", re.IGNORECASE)


@mcp.tool()
def write_confirmed_reservation(
    api_key: str,
    reservation_id: int,
    customer_name: str,
    customer_surname: str,
    car_number: str,
    start_datetime: str,
    end_datetime: str,
    approval_time: str,
) -> str:
    """Append a confirmed reservation to the log file.

    Entry format:
        Name Surname | CAR-NUM | YYYY-MM-DD HH:MM – YYYY-MM-DD HH:MM | Approved: YYYY-MM-DD HH:MM:SS

    Args:
        api_key:          Shared secret that authorises the write.
        reservation_id:   Database reservation ID (used in the log response only).
        customer_name:    Customer first name.
        customer_surname: Customer last name.
        car_number:       Vehicle registration plate (2-12 alphanumerics / hyphens).
        start_datetime:   Reservation start in ISO 8601 format.
        end_datetime:     Reservation end in ISO 8601 format.
        approval_time:    Timestamp when admin approved (ISO 8601).

    Returns:
        Confirmation string on success.

    Raises:
        PermissionError: api_key does not match the server secret.
        ValueError:      Any field fails validation.
        RuntimeError:    Server is misconfigured (no API key set).
    """
    # --- auth ---
    if not _API_KEY:
        raise RuntimeError("MCP_API_KEY is not configured on the server.")
    if not secrets.compare_digest(api_key, _API_KEY):
        raise PermissionError("Invalid API key.")

    # --- field validation ---
    if not customer_name.strip() or not customer_surname.strip():
        raise ValueError("customer_name and customer_surname must not be empty.")

    if not _CAR_NUMBER_RE.match(car_number):
        raise ValueError(
            f"Invalid car_number {car_number!r}. "
            "Expected 2-12 alphanumeric characters or hyphens."
        )

    try:
        start = datetime.fromisoformat(start_datetime)
        end = datetime.fromisoformat(end_datetime)
        approval = datetime.fromisoformat(approval_time)
    except ValueError as exc:
        raise ValueError(f"Bad datetime value: {exc}") from exc

    if end <= start:
        raise ValueError("end_datetime must be after start_datetime.")

    # --- format entry ---
    name = f"{customer_name.strip()} {customer_surname.strip()}"
    period = (
        f"{start.strftime('%Y-%m-%d %H:%M')} \u2013 {end.strftime('%Y-%m-%d %H:%M')}"
    )
    approved_str = approval.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{name} | {car_number.upper()} | {period} | Approved: {approved_str}\n"

    # --- write ---
    _RESERVATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESERVATIONS_FILE, "a", encoding="utf-8") as fh:
        fh.write(line)

    logger.info("Reservation #%d written to log: %s", reservation_id, line.rstrip())
    return f"Reservation #{reservation_id} written to log."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8520"))
    logger.info("Starting CityPark MCP server on %s:%d", host, port)
    mcp.run(transport="sse", host=host, port=port)
