"""
CityPark Admin API — Reservation decision interface for administrators.

Reservations are created by the chatbot booking agent; admins only approve or reject.

Endpoints:
    GET    /api/reservations               — list reservations (filter by status)
    GET    /api/reservations/{id}          — get a single reservation
    POST   /api/reservations/{id}/approve  — approve a pending reservation
    POST   /api/reservations/{id}/reject   — reject a pending reservation
"""
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import secrets

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database as db
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CityPark Admin API",
    description="Admin service for managing parking reservations.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
    """Validate admin credentials via HTTP Basic Auth."""
    settings = get_settings()
    correct_username = secrets.compare_digest(credentials.username, settings.admin_username)
    correct_password = secrets.compare_digest(credentials.password, settings.admin_password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ReservationResponse(BaseModel):
    id: int
    space_id: int
    floor: str
    space_number: str
    customer_name: str
    customer_surname: str
    car_number: str
    start_datetime: datetime
    end_datetime: datetime
    total_cost: float
    status: str
    created_at: datetime


class StatusResponse(BaseModel):
    message: str
    reservation: ReservationResponse


class ReservationStatus(str, Enum):
    pending = "pending"
    confirmed = "confirmed"
    cancelled = "cancelled"
    completed = "completed"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/reservations", response_model=list[ReservationResponse])
def list_reservations(
    status: Optional[ReservationStatus] = Query(None),
    _: str = Depends(verify_admin),
):
    """List reservations, optionally filtered by status."""
    rows = db.list_reservations(status.value if status else None)
    return [ReservationResponse(**r) for r in rows]


@app.get("/api/reservations/{reservation_id}", response_model=ReservationResponse)
def get_reservation(reservation_id: int, _: str = Depends(verify_admin)):
    """Get a single reservation by ID."""
    row = db.get_reservation(reservation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Reservation not found")
    return ReservationResponse(**row)


@app.post("/api/reservations/{reservation_id}/approve", response_model=StatusResponse)
def approve_reservation(reservation_id: int, _: str = Depends(verify_admin)):
    """Approve a pending reservation (set status to confirmed)."""
    row = db.get_reservation(reservation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Reservation not found")
    if row["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Reservation is '{row['status']}', only 'pending' reservations can be approved.",
        )

    updated = db.update_reservation_status(reservation_id, "confirmed")
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update reservation status")

    logger.info("Reservation #%d approved", reservation_id)
    return StatusResponse(
        message=f"Reservation #{reservation_id} has been approved.",
        reservation=ReservationResponse(**updated),
    )


@app.post("/api/reservations/{reservation_id}/reject", response_model=StatusResponse)
def reject_reservation(reservation_id: int, _: str = Depends(verify_admin)):
    """Reject a pending reservation (set status to cancelled)."""
    row = db.get_reservation(reservation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Reservation not found")
    if row["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Reservation is '{row['status']}', only 'pending' reservations can be rejected.",
        )

    updated = db.update_reservation_status(reservation_id, "cancelled")
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update reservation status")

    logger.info("Reservation #%d rejected", reservation_id)
    return StatusResponse(
        message=f"Reservation #{reservation_id} has been rejected.",
        reservation=ReservationResponse(**updated),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin UI
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def admin_ui():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
