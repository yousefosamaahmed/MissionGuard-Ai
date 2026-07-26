# 5.0.0 — Global Challenge UI/UX Experience (2026-07-22)

- Rebuilt the first impression as a cinematic MissionGuard Launchpad inspired by the supplied SpaceY Figma reference.
- Added a clear project explanation and START MISSION CONTROL call to action.
- Added numbered, dedicated workspaces for every product capability.
- Added responsive feature, workflow, scientific-proof, and live mission metric sections.
- Added a professional Team & Contact page with the supplied portraits, biographies, skills, and email links.
- Preserved all existing OPS-SAT machine-learning, PostgreSQL, pgAdmin, Docker, upload, incident, and reporting workflows.
- Kept Dark and Light appearance support.
- Passed the complete automated test suite: 13/13.

# Runtime hotfix (2026-07-20)

- Added an explicit modern `typing_extensions` requirement so `psycopg` can import `TypeVar`.
- Changed the local pgAdmin login email to `admin@missionguard.com`; `.local` is rejected by pgAdmin 9.
- Corrected the Windows startup script so normal Docker Compose progress text is not mistaken for a non-zero exit code.
- Added a Docker build-time import smoke test for `psycopg`.

## Windows Docker port-fix build

- Changed the Windows-published PostgreSQL port from `5432` to `55432` to avoid local/reserved port conflicts.
- Kept the internal Docker PostgreSQL endpoint unchanged at `postgres:5432`.
- Improved `START_DOCKER_WINDOWS.bat` with Docker-engine checks, cleanup of stale containers, readiness waiting, service status, diagnostics, and automatic browser opening.

# Changelog

## 4.0.2 — Streamlit Compatibility Hotfix

- Fixed `TypeError: HtmlMixin.html() got an unexpected keyword argument 'width'`.
- Added runtime Streamlit API detection so the same project works with the declared minimum Streamlit 1.42 and newer releases.
- Restored the legacy Components HTML bridge only when the installed Streamlit version requires it.
- Made full-width charts, tables, buttons, and downloads select the correct argument for the installed Streamlit version.
- Re-ran all 12 automated tests and rendered every workspace page in Dark and Light modes under Streamlit 1.42 without exceptions.

## 4.0.1 — Interface Layout and Compatibility Repair

- Added consistent Plotly chart containers, heights, margins, and responsive spacing.
- Prevented chart titles, section headings, captions, tables, and the footer from overlapping.
- Improved axis auto-margins so long feature and channel labels remain visible.
- Added mobile-width spacing adjustments for metrics, charts, and the hero section.
- Initially migrated layout elements to the newer Streamlit width API; superseded by the cross-version compatibility layer in 4.0.2.
- Initially migrated the theme bridge to the newer `st.html` API; superseded by runtime API detection in 4.0.2.
- Verified every workspace page in both Dark and Light modes.
- Verified the packaged real CSV upload workflow and all automated tests.

## 4.0.0 — OPSSAT Reliability Upgrade

- Added strict raw and engineered CSV upload validation.
- Added duplicate removal, invalid-row checks, binary-label validation, and unknown-channel warnings.
- Added packaged Isolation Forest and Random Forest bundles plus feature-schema metadata.
- Added explicit internal train/validation and official test feature files.
- Added ground-truth segment metrics for labeled uploads.
- Added event-based evaluation and event detection ledger.
- Added telemetry distribution-drift monitoring against nominal training data.
- Replaced heuristic confidence wording with a transparent decision-margin index.
- Added a full dataset card and expanded model limitations.
- Added real normal, anomaly, recovery-sequence, magnetometer, and photodiode upload samples.
- Added tests for uploads, duplicate handling, evaluation, events, and drift.
## Windows Docker launcher fix

- Fixed false startup failures caused by Windows PowerShell 5.1 treating Docker Compose progress written to stderr (for example, `Container ... Stopping`) as a terminating error.
- Docker output is now captured with `Start-Process`, and success/failure is determined only from Docker's numeric exit code.

