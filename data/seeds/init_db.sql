-- =============================================================================
-- CityPark Premium Parking - Database Initialisation
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Parking spaces
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parking_spaces (
    id           SERIAL PRIMARY KEY,
    floor        VARCHAR(10)  NOT NULL,
    space_number VARCHAR(10)  NOT NULL,
    space_type   VARCHAR(20)  NOT NULL
        CHECK (space_type IN ('standard', 'compact', 'handicapped', 'ev')),
    status       VARCHAR(20)  NOT NULL DEFAULT 'operating'
        CHECK (status IN ('operating', 'maintenance')),
    UNIQUE (floor, space_number)
);

-- ---------------------------------------------------------------------------
-- 2. Prices
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices (
    id          SERIAL PRIMARY KEY,
    price_type  VARCHAR(30)     NOT NULL UNIQUE,
    amount      NUMERIC(10, 2)  NOT NULL,
    currency    CHAR(3)         NOT NULL DEFAULT 'EUR',
    description TEXT
);

-- ---------------------------------------------------------------------------
-- 3. Reservations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reservations (
    id                SERIAL PRIMARY KEY,
    space_id          INTEGER         REFERENCES parking_spaces(id),
    customer_name     VARCHAR(100)    NOT NULL,
    customer_surname  VARCHAR(100)    NOT NULL,
    car_number        VARCHAR(20)     NOT NULL,
    start_datetime    TIMESTAMP       NOT NULL,
    end_datetime      TIMESTAMP       NOT NULL,
    status            VARCHAR(20)     NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'confirmed', 'cancelled', 'completed')),
    total_cost        NUMERIC(10, 2),
    created_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_period CHECK (start_datetime < end_datetime)
);

-- ---------------------------------------------------------------------------
-- 4. Working hours
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS working_hours (
    id           SERIAL PRIMARY KEY,
    day_of_week  VARCHAR(10),
    open_time    TIME,
    close_time   TIME,
    is_24h       BOOLEAN NOT NULL DEFAULT TRUE
);

-- =============================================================================
-- Seed data
-- =============================================================================

-- ── Working hours (24/7 automated facility) ──────────────────────────────────
INSERT INTO working_hours (day_of_week, is_24h) VALUES
    ('Monday',    TRUE),
    ('Tuesday',   TRUE),
    ('Wednesday', TRUE),
    ('Thursday',  TRUE),
    ('Friday',    TRUE),
    ('Saturday',  TRUE),
    ('Sunday',    TRUE)
ON CONFLICT DO NOTHING;

-- ── Pricing tiers ────────────────────────────────────────────────────────────
INSERT INTO prices (price_type, amount, currency, description) VALUES
    ('hourly',      3.00, 'EUR', 'Standard hourly rate for all space types'),
    ('daily_max',  25.00, 'EUR', 'Maximum charge per calendar day'),
    ('monthly',   200.00, 'EUR', 'Monthly subscription — unlimited in/out access'),
    ('ev_charging', 0.30, 'EUR', 'EV charging rate per kWh')
ON CONFLICT (price_type) DO NOTHING;

-- ── Parking spaces (150 total across 3 floors) ───────────────────────────────
-- Floor B1: 10 handicapped, 10 EV, 30 standard
DO $$
DECLARE i INT;
BEGIN
    FOR i IN 1..10 LOOP
        INSERT INTO parking_spaces (floor, space_number, space_type)
        VALUES ('B1', 'H' || LPAD(i::TEXT, 2, '0'), 'handicapped')
        ON CONFLICT DO NOTHING;
    END LOOP;

    FOR i IN 1..10 LOOP
        INSERT INTO parking_spaces (floor, space_number, space_type)
        VALUES ('B1', 'EV' || LPAD(i::TEXT, 2, '0'), 'ev')
        ON CONFLICT DO NOTHING;
    END LOOP;

    FOR i IN 1..30 LOOP
        INSERT INTO parking_spaces (floor, space_number, space_type)
        VALUES ('B1', 'S' || LPAD(i::TEXT, 2, '0'), 'standard')
        ON CONFLICT DO NOTHING;
    END LOOP;
END $$;

-- Floor B2: 20 compact, 30 standard
DO $$
DECLARE i INT;
BEGIN
    FOR i IN 1..20 LOOP
        INSERT INTO parking_spaces (floor, space_number, space_type)
        VALUES ('B2', 'C' || LPAD(i::TEXT, 2, '0'), 'compact')
        ON CONFLICT DO NOTHING;
    END LOOP;

    FOR i IN 1..30 LOOP
        INSERT INTO parking_spaces (floor, space_number, space_type)
        VALUES ('B2', 'S' || LPAD(i::TEXT, 2, '0'), 'standard')
        ON CONFLICT DO NOTHING;
    END LOOP;
END $$;

-- Floor B3: 10 EV, 40 standard
DO $$
DECLARE i INT;
BEGIN
    FOR i IN 11..20 LOOP
        INSERT INTO parking_spaces (floor, space_number, space_type)
        VALUES ('B3', 'EV' || LPAD(i::TEXT, 2, '0'), 'ev')
        ON CONFLICT DO NOTHING;
    END LOOP;

    FOR i IN 1..40 LOOP
        INSERT INTO parking_spaces (floor, space_number, space_type)
        VALUES ('B3', 'S' || LPAD(i::TEXT, 2, '0'), 'standard')
        ON CONFLICT DO NOTHING;
    END LOOP;
END $$;
