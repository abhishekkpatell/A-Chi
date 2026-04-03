---
title: SupportDesk OpenEnv
emoji: 📬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
  - customer-support
  - triage
---

# SupportDesk OpenEnv

SupportDesk OpenEnv is a real-world support-operations simulation for training agents on customer ticket triage, billing resolution, and incident coordination. The agent sees a realistic case summary, policy constraints, and a small number of turns to take corrective actions. Each episode is scored with a deterministic rubric and returns dense partial credit as the agent improves.

## Why this environment exists

Support teams spend most of their time routing requests, drafting compliant replies, and escalating incidents without violating policy. This environment simulates that workflow in a compact, reproducible form that is suitable for agent training and evaluation.

## Tasks

The environment ships with three tasks:

1. **Easy — Password-reset login lockout triage**
   - Route the ticket to the account-access queue.
   - Mark it as medium priority.
   - Provide a safe response that references the reset link, spam folder, and 2FA.

2. **Medium — Duplicate billing charge resolution**
   - Route to billing.
   - Mark as high priority.
   - Confirm that the refund goes back to the original payment method and set expectations for a 3–5 business day processing window.

3. **Hard — EU outage incident coordination**
   - Assign incident management ownership.
   - Mark the outage urgent and set the correct severity.
   - Draft a customer-facing update that mentions the status page, a workaround, and explicitly avoids promising an exact ETA.

## Action space

`SupportAction` is a Pydantic model with these fields:

- `action_type`: `classify | draft_reply | escalate | resolve | finalize`
- `issue_type`: issue label
- `priority`: `low | medium | high | urgent`
- `team`: routing target
- `severity`: `sev4 | sev3 | sev2 | sev1`
- `status`: case status update
- `tags`: list of case tags
- `message`: customer-facing or internal text
- `internal_note`: handoff note
- `refund_amount`: optional amount to refund
- `confidence`: float from 0.0 to 1.0

## Observation space

`SupportObservation` contains:

- current task metadata
- turn counter and remaining turns
- case summary
- open questions and constraints
- last feedback
- current best score
- whether the episode is done

## Reward design

The environment provides dense reward through a deterministic rubric:

- Correct routing fields raise the score.
- Required policy phrases add partial credit.
- Forbidden phrases and unsafe behavior are penalized.
- Rewards are based on improvement over the best previous score, so agents get credit for incremental progress.

## Setup

```bash
pip install -e .
uvicorn supportdesk_env.server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t supportdesk-openenv .
docker run --rm -p 8000:8000 supportdesk-openenv
```

## API

- `POST /reset` — start a task episode
- `POST /step` — submit a `SupportAction`
- `GET /state` — inspect the current internal state
- `GET /health` — liveness check
- `GET /tasks` — list available tasks

## Baseline

The repository includes `inference.py`, which uses the OpenAI Python client and the following environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `OPENAI_API_KEY`)

For local smoke testing, it also supports `--offline`.

Reference offline scores:

- Easy: 1.00
- Medium: 1.00
- Hard: 1.00
- Mean: 1.00

Start the environment first, then generate the same scores locally with:

```bash
python inference.py --offline
```

The exact model-based baseline depends on the selected model, but the script prints per-task scores and a mean score in a reproducible `[START]`, `[STEP]`, `[END]` log format.

## Project layout

```text
.
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── supportdesk_env/
│   ├── __init__.py
│   ├── client.py
│   ├── grader.py
│   ├── models.py
│   ├── task_bank.py
│   └── server/
│       ├── __init__.py
│       ├── app.py
│       └── environment.py
├── scripts/
│   └── validate.py
└── tests/
    └── test_environment.py
```
