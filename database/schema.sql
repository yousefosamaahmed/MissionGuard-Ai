-- MissionGuard AI core PostgreSQL schema.
-- The bootstrap script replaces __SCHEMA__ with POSTGRES_SCHEMA.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS __SCHEMA__;

CREATE TABLE IF NOT EXISTS __SCHEMA__.missions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    mission_code TEXT UNIQUE,
    spacecraft_name TEXT,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS __SCHEMA__.datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    dataset_code TEXT UNIQUE,
    source_type TEXT NOT NULL,
    source_organization TEXT,
    source_url TEXT,
    license_name TEXT,
    description TEXT,
    version TEXT,
    row_count BIGINT NOT NULL DEFAULT 0,
    feature_count INTEGER NOT NULL DEFAULT 0,
    is_labeled BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS __SCHEMA__.dataset_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL REFERENCES __SCHEMA__.datasets(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_role TEXT NOT NULL,
    file_path TEXT NOT NULL,
    storage_provider TEXT NOT NULL DEFAULT 'local',
    file_size_bytes BIGINT,
    mime_type TEXT,
    sha256_hash TEXT,
    row_count BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_dataset_files_dataset_id
    ON __SCHEMA__.dataset_files(dataset_id);

CREATE TABLE IF NOT EXISTS __SCHEMA__.telemetry_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mission_id UUID REFERENCES __SCHEMA__.missions(id) ON DELETE SET NULL,
    dataset_id UUID REFERENCES __SCHEMA__.datasets(id) ON DELETE SET NULL,
    session_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_file_name TEXT,
    sampling_interval_seconds DOUBLE PRECISION,
    total_samples BIGINT NOT NULL DEFAULT 0,
    validation_status TEXT NOT NULL DEFAULT 'pending',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_telemetry_session_source
        CHECK (mission_id IS NOT NULL OR dataset_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS ix_telemetry_sessions_mission_id
    ON __SCHEMA__.telemetry_sessions(mission_id);
CREATE INDEX IF NOT EXISTS ix_telemetry_sessions_dataset_id
    ON __SCHEMA__.telemetry_sessions(dataset_id);
CREATE INDEX IF NOT EXISTS ix_telemetry_sessions_created_at
    ON __SCHEMA__.telemetry_sessions(created_at DESC);

CREATE TABLE IF NOT EXISTS __SCHEMA__.telemetry_samples (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES __SCHEMA__.telemetry_sessions(id) ON DELETE CASCADE,
    mission_phase_id UUID,
    sample_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ,
    segment_identifier TEXT,
    split_type TEXT NOT NULL DEFAULT 'upload',
    ground_truth_label BOOLEAN,
    anomaly_type TEXT,
    sample_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_telemetry_sample_index UNIQUE (session_id, sample_index)
);

CREATE INDEX IF NOT EXISTS ix_telemetry_samples_session_id
    ON __SCHEMA__.telemetry_samples(session_id);

CREATE TABLE IF NOT EXISTS __SCHEMA__.feature_vectors (
    id BIGSERIAL PRIMARY KEY,
    telemetry_sample_id BIGINT NOT NULL REFERENCES __SCHEMA__.telemetry_samples(id) ON DELETE CASCADE,
    schema_name TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    feature_values JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_sample_feature_schema
        UNIQUE (telemetry_sample_id, schema_name, schema_version)
);

CREATE INDEX IF NOT EXISTS ix_feature_vectors_sample_id
    ON __SCHEMA__.feature_vectors(telemetry_sample_id);

CREATE TABLE IF NOT EXISTS __SCHEMA__.data_quality_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES __SCHEMA__.telemetry_sessions(id) ON DELETE CASCADE,
    row_count BIGINT NOT NULL DEFAULT 0,
    invalid_timestamps INTEGER NOT NULL DEFAULT 0,
    duplicate_timestamps INTEGER NOT NULL DEFAULT 0,
    long_missing_gaps INTEGER NOT NULL DEFAULT 0,
    constant_sensors JSONB NOT NULL DEFAULT '[]'::jsonb,
    out_of_domain_values JSONB NOT NULL DEFAULT '{}'::jsonb,
    missing_value_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    sampling_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation_messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    overall_status TEXT NOT NULL DEFAULT 'valid',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_data_quality_session UNIQUE (session_id)
);

CREATE TABLE IF NOT EXISTS __SCHEMA__.model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT,
    training_dataset_id UUID REFERENCES __SCHEMA__.datasets(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'training',
    feature_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    training_parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_size_bytes BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_name_version UNIQUE (model_name, version)
);

CREATE TABLE IF NOT EXISTS __SCHEMA__.model_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id UUID NOT NULL REFERENCES __SCHEMA__.model_versions(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    storage_provider TEXT NOT NULL DEFAULT 'local',
    file_size_bytes BIGINT,
    sha256_hash TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_artifact
        UNIQUE (model_version_id, artifact_type, file_path)
);

CREATE INDEX IF NOT EXISTS ix_model_artifacts_model_version_id
    ON __SCHEMA__.model_artifacts(model_version_id);

CREATE TABLE IF NOT EXISTS __SCHEMA__.model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id UUID NOT NULL REFERENCES __SCHEMA__.model_versions(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES __SCHEMA__.datasets(id) ON DELETE SET NULL,
    evaluation_type TEXT NOT NULL,
    split_name TEXT,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    accuracy_score DOUBLE PRECISION,
    mcc_score DOUBLE PRECISION,
    roc_auc DOUBLE PRECISION,
    pr_auc DOUBLE PRECISION,
    false_alarm_rate DOUBLE PRECISION,
    false_alarms_per_1000 DOUBLE PRECISION,
    mean_detection_delay DOUBLE PRECISION,
    median_detection_delay DOUBLE PRECISION,
    event_precision DOUBLE PRECISION,
    event_recall DOUBLE PRECISION,
    event_f1 DOUBLE PRECISION,
    detected_events INTEGER,
    missed_events INTEGER,
    false_event_alerts INTEGER,
    duplicate_alerts INTEGER,
    confusion_matrix JSONB NOT NULL DEFAULT '{}'::jsonb,
    extra_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_model_metrics_model_version_id
    ON __SCHEMA__.model_metrics(model_version_id);
CREATE INDEX IF NOT EXISTS ix_model_metrics_dataset_id
    ON __SCHEMA__.model_metrics(dataset_id);

CREATE TABLE IF NOT EXISTS __SCHEMA__.analysis_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES __SCHEMA__.telemetry_sessions(id) ON DELETE CASCADE,
    model_version_id UUID NOT NULL REFERENCES __SCHEMA__.model_versions(id) ON DELETE RESTRICT,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    total_predictions INTEGER NOT NULL DEFAULT 0,
    total_anomalies INTEGER NOT NULL DEFAULT 0,
    total_incidents INTEGER NOT NULL DEFAULT 0,
    mission_health_score DOUBLE PRECISION,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_analysis_runs_session_model
    ON __SCHEMA__.analysis_runs(session_id, model_version_id, created_at DESC);

CREATE TABLE IF NOT EXISTS __SCHEMA__.predictions (
    id BIGSERIAL PRIMARY KEY,
    analysis_run_id UUID NOT NULL REFERENCES __SCHEMA__.analysis_runs(id) ON DELETE CASCADE,
    telemetry_sample_id BIGINT NOT NULL REFERENCES __SCHEMA__.telemetry_samples(id) ON DELETE CASCADE,
    predicted_anomaly BOOLEAN NOT NULL,
    risk_level TEXT NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    isolation_score DOUBLE PRECISION,
    forecast_residual_score DOUBLE PRECISION,
    rule_score DOUBLE PRECISION,
    persistence_score DOUBLE PRECISION,
    early_warning_score DOUBLE PRECISION,
    top_feature TEXT,
    out_of_distribution BOOLEAN NOT NULL DEFAULT FALSE,
    human_review_required BOOLEAN NOT NULL DEFAULT FALSE,
    explanation TEXT,
    feature_contributions JSONB NOT NULL DEFAULT '{}'::jsonb,
    rule_violations JSONB NOT NULL DEFAULT '[]'::jsonb,
    affected_subsystems JSONB NOT NULL DEFAULT '[]'::jsonb,
    prediction_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_analysis_sample_prediction
        UNIQUE (analysis_run_id, telemetry_sample_id)
);

CREATE INDEX IF NOT EXISTS ix_predictions_analysis_run_id
    ON __SCHEMA__.predictions(analysis_run_id);

CREATE TABLE IF NOT EXISTS __SCHEMA__.incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_run_id UUID NOT NULL REFERENCES __SCHEMA__.analysis_runs(id) ON DELETE CASCADE,
    incident_code TEXT NOT NULL,
    start_sample_id BIGINT REFERENCES __SCHEMA__.telemetry_samples(id) ON DELETE SET NULL,
    end_sample_id BIGINT REFERENCES __SCHEMA__.telemetry_samples(id) ON DELETE SET NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    duration_samples INTEGER,
    severity TEXT NOT NULL,
    peak_risk_score DOUBLE PRECISION NOT NULL,
    peak_confidence DOUBLE PRECISION,
    top_feature TEXT,
    affected_subsystems JSONB NOT NULL DEFAULT '[]'::jsonb,
    human_review_required BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'open',
    summary TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_incident_code UNIQUE (incident_code)
);

CREATE INDEX IF NOT EXISTS ix_incidents_analysis_run_id
    ON __SCHEMA__.incidents(analysis_run_id);
CREATE INDEX IF NOT EXISTS ix_incidents_status
    ON __SCHEMA__.incidents(status);
