# Limitations

1. OPSSAT-AD covers nine selected channels rather than the complete spacecraft telemetry set.
2. Predictions operate at manually defined segment level.
3. Engineered-feature evidence explains statistical deviation, not causal hardware diagnosis.
4. Binary labels do not identify anomaly subtype or engineering root cause.
5. Performance may change on different missions, unseen channels, sampling policies, or distribution shifts.
6. Drift estimates are less stable for very small uploads.
7. The supervised anomaly score is uncalibrated and must not be interpreted as an operational failure probability.
8. Human review is required for operational interpretation.
