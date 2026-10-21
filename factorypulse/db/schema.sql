-- ============================================================
-- MaintenAI— TimescaleDB Production Telemetry Schema
-- ============================================================

-- Ensure pgcrypto / uuid-ossp for random UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- 1. Machines Directory
CREATE TABLE IF NOT EXISTS machines (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    equipment_type VARCHAR(100) NOT NULL DEFAULT 'turbofan',
    location VARCHAR(200) DEFAULT 'Factory Floor 1',
    status VARCHAR(50) NOT NULL DEFAULT 'healthy',
    installed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata_json JSONB DEFAULT '{}'::jsonb
);

-- 2. Sensor Readings (TimescaleDB Hypertable)
CREATE TABLE IF NOT EXISTS sensor_readings (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    machine_id VARCHAR(50) NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    temperature DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    vibration DOUBLE PRECISION,
    rpm DOUBLE PRECISION,
    current DOUBLE PRECISION,
    operating_cycle INTEGER,
    ingestion_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(100) NOT NULL DEFAULT 'SYNTHETIC / DEVELOPMENT',
    validation_metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (machine_id, timestamp)
);

-- Convert sensor_readings to TimescaleDB hypertable partitioned by timestamp
SELECT create_hypertable('sensor_readings', 'timestamp', if_not_exists => TRUE);

-- Optimized compound indexes for fast recent-window slicing & historical filtering
CREATE INDEX IF NOT EXISTS ix_sensor_readings_machine_time
    ON sensor_readings (machine_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS ix_sensor_readings_time
    ON sensor_readings (timestamp DESC);

-- 3. Model Predictions History
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    machine_id VARCHAR(50) NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    anomaly_status VARCHAR(50) NOT NULL DEFAULT 'NORMAL',
    predicted_rul_cycles DOUBLE PRECISION NOT NULL,
    failure_probability DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    probable_failure_modes JSONB DEFAULT '{}'::jsonb,
    model_versions JSONB DEFAULT '{}'::jsonb,
    feature_version VARCHAR(50) DEFAULT 'v1.0',
    latency_ms DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_predictions_machine_time
    ON predictions (machine_id, timestamp DESC);

-- 4. Alerts History & Audit
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(50) PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    threshold VARCHAR(100),
    model_version VARCHAR(100),
    acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_alerts_machine_time
    ON alerts (machine_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS ix_alerts_unack
    ON alerts (acknowledged, severity);

-- 5. Seed Core Fleet Assets (Idempotent)
INSERT INTO machines (id, name, equipment_type, status, location)
VALUES
    ('M001', 'High-Load Primary Turbofan Engine M001', 'turbofan', 'healthy', 'Test Cell Alpha'),
    ('turbofan-001', 'CF6 Turbofan Engine #1', 'turbofan', 'critical', 'Test Cell 1'),
    ('turbofan-002', 'CF6 Turbofan Engine #2', 'turbofan', 'healthy', 'Test Cell 2'),
    ('bearing-004', 'Rexnord ZA-2115 High-Speed Bearing', 'bearing', 'warning', 'Shaft Bay 4'),
    ('bearing-005', 'SKF Explorer Spherical Roller Bearing', 'bearing', 'healthy', 'Shaft Bay 5'),
    ('cnc-012', 'DMG Mori 5-Axis CNC Milling Center', 'cnc_mill', 'warning', 'Machine Shop 1'),
    ('cnc-013', 'Mazak Variaxis C-600 CNC Machine', 'cnc_mill', 'healthy', 'Machine Shop 2')
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    equipment_type = EXCLUDED.equipment_type,
    location = EXCLUDED.location;
