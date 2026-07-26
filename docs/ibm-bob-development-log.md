# IBM Bob Development Log

This file is a submission-ready evidence template. Replace the example commit references with the actual Git commit hashes produced during development.

| Date | Task | Bob Workflow | Prompt / Request | Output Used | Human Validation | Commit |
|---|---|---|---|---|---|---|
| 2026-07-16 | Architecture | Plan | Design a modular spacecraft telemetry anomaly-detection architecture with validation, temporal features, scoring, explanations, tests, and Streamlit UI. | Proposed module boundaries and data flow. | Removed unnecessary services and constrained the MVP. | `ADD_COMMIT` |
| 2026-07-16 | Data validation | Code | Implement robust CSV validation for timestamps, duplicates, missing gaps, physical domains, and constant sensors. | Initial validation module. | Added limited interpolation and explicit long-gap warnings. | `ADD_COMMIT` |
| 2026-07-16 | Model pipeline | Code | Separate nominal training, validation calibration, and independent testing. | Isolation Forest and forecast-residual pipeline. | Reviewed leakage controls and score ranges. | `ADD_COMMIT` |
| 2026-07-16 | Explainability | Plan + Code | Rank sensor evidence, generate root-cause hypotheses, alternatives, and transparent counterfactuals. | Incident intelligence module. | Reworded causal claims as hypotheses. | `ADD_COMMIT` |
| 2026-07-16 | Reliability tests | Test | Create pytest coverage for validation, rules, model outputs, fault injection, and incident intelligence. | Automated test suite. | Ran tests from a clean project environment. | `ADD_COMMIT` |
| 2026-07-16 | Security and safety | Review | Review file upload, credentials, LLM grounding, and autonomous-action risks. | Risk findings and guardrails. | Kept credentials in environment variables and prohibited commands. | `ADD_COMMIT` |
| 2026-07-16 | Documentation | Docs | Produce README, architecture, model card, data card, limitations, and demo script. | Submission documentation. | Cross-checked claims against implemented code. | `ADD_COMMIT` |

## Evidence to Capture Before Submission

- Screenshots of Bob planning and code-review sessions.
- Git commits corresponding to the rows above.
- A short paragraph explaining which Bob output was rejected or modified and why.
- Test command output.
- Final architecture review and demo rehearsal notes.
