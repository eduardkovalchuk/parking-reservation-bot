"""
CityPark Admin API — Reservation management for administrators.

Endpoints:
    POST   /api/reservations              — create a new reservation (pending)
    GET    /api/reservations               — list reservations (filter by status)
    GET    /api/reservations/{id}          — get a single reservation
    POST   /api/reservations/{id}/approve  — approve a pending reservation
    POST   /api/reservations/{id}/reject   — reject a pending reservation
"""
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

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

_api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """Validate the API key from the X-API-Key header."""
    expected = get_settings().admin_api_key
    if not expected:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured on server")
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SpaceType(str, Enum):
    standard = "standard"
    compact = "compact"
    handicapped = "handicapped"
    ev = "ev"


class ReservationCreate(BaseModel):
    customer_name: str = Field(..., min_length=1, max_length=100)
    customer_surname: str = Field(..., min_length=1, max_length=100)
    car_number: str = Field(..., min_length=1, max_length=20)
    start_datetime: datetime
    end_datetime: datetime
    space_type: SpaceType = SpaceType.standard
    space_id: Optional[int] = Field(None, description="Specific parking space ID. If provided, skips auto-assignment.")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_name": "John",
                "customer_surname": "Doe",
                "car_number": "ABC-1234",
                "start_datetime": "2026-04-15T09:00:00",
                "end_datetime": "2026-04-15T17:00:00",
                "space_type": "standard",
            }
        }


class ReservationResponse(BaseModel):
    id: int
    space_id: int
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

@app.post("/api/reservations", response_model=ReservationResponse, status_code=201)
def create_reservation(body: ReservationCreate):
    """Create a new reservation in pending state."""
    if body.end_datetime <= body.start_datetime:
        raise HTTPException(status_code=400, detail="end_datetime must be after start_datetime")

    if body.space_id is not None:
        space = db.get_space(body.space_id)
        if not space:
            raise HTTPException(status_code=404, detail=f"Parking space {body.space_id} not found")
        if space["status"] != "operating":
            raise HTTPException(
                status_code=409,
                detail=f"Parking space {body.space_id} is '{space['status']}', not available.",
            )
        if not db.is_space_available(body.space_id, body.start_datetime, body.end_datetime):
            raise HTTPException(
                status_code=409,
                detail=f"Parking space {body.space_id} is already reserved for the requested period.",
            )
    else:
        space = db.find_available_space(body.space_type.value, body.start_datetime, body.end_datetime)
        if not space:
            raise HTTPException(
                status_code=409,
                detail=f"No available {body.space_type.value} spaces for the requested period.",
            )

    total_cost = db._calculate_cost(body.start_datetime, body.end_datetime)

    reservation = db.create_reservation(
        space_id=space["id"],
        customer_name=body.customer_name,
        customer_surname=body.customer_surname,
        car_number=body.car_number,
        start=body.start_datetime,
        end=body.end_datetime,
        total_cost=total_cost,
    )

    logger.info("Created reservation #%d (pending)", reservation["id"])
    return ReservationResponse(**reservation)


@app.get("/api/reservations", response_model=list[ReservationResponse])
def list_reservations(status: Optional[ReservationStatus] = Query(None)):
    """List reservations, optionally filtered by status."""
    rows = db.list_reservations(status.value if status else None)
    return [ReservationResponse(**r) for r in rows]


@app.get("/api/reservations/{reservation_id}", response_model=ReservationResponse)
def get_reservation(reservation_id: int):
    """Get a single reservation by ID."""
    row = db.get_reservation(reservation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Reservation not found")
    return ReservationResponse(**row)


@app.post("/api/reservations/{reservation_id}/approve", response_model=StatusResponse)
def approve_reservation(reservation_id: int, _: str = Depends(verify_api_key)):
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
def reject_reservation(reservation_id: int, _: str = Depends(verify_api_key)):
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
