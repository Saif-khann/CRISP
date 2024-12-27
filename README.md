<h1 align="center">CRISP</h1>

<p align="center">
  <strong>Construction Recognition &amp; Intelligence for Stage Progress</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Flask-2.3%2B-000000?logo=flask&logoColor=white" alt="Flask 2.3+">
  <img src="https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.15+">
  <img src="https://img.shields.io/badge/tests-44%20passing-3FB950" alt="44 tests passing">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT licence">
</p>

<p align="center">
  <em>Reads a construction site photograph, determines which phase of<br>
  construction it shows, and derives project completion from that evidence.</em>
</p>

---

A three-model CNN ensemble classifies the broad construction phase. A
stage-specific model then identifies the exact sub-stage within it. A
computer-vision pipeline compares two photographs of the same site to
highlight what has physically been built between them.

Completion percentage is never typed in by a human. It is computed from
validation history against a weighted stage model, so every number on the
dashboard traces back to a specific photograph.

<table>
<tr><td width="33%" valign="top">

**211 ms**
<sub>full validation, 4 forward passes</sub>

</td><td width="33%" valign="top">

**100% / 0.1%**
<sub>structural change recall / false positive on lighting-only change</sub>

</td><td width="33%" valign="top">

**44**
<sub>unit tests, plus 69 live integration checks</sub>

</td></tr>
</table>

### Contents

[Features](#features) · [Architecture](#architecture) ·
[Construction model](#construction-model) ·
[ML pipeline](#machine-learning-pipeline)
---

## Features

| | |
|---|---|
| **Stage validation** | A worker uploads a photograph and declares the stage it should show. The ensemble classifies it independently. Disagreement rejects the upload with an explanation. This is the integrity mechanism: a project cannot be marked further along than its photographic evidence supports. Deliberate overrides are permitted and recorded. |
| **Progress tracking** | Completion is derived from validation history against a weighted stage model, never entered by hand. |
| **Milestone comparison** | Any two recorded validations produce a plain-language progress delta and a breakdown of which stages completed in between. |
| **Visual change analysis** | Two photographs of the same site are aligned, exposure-normalised and compared structurally, highlighting genuine construction change only. Expert accounts only. |
| **Audit reports** | Any validation exports as a PDF containing the photograph, classification, confidence figures and progress state. |
| **Site map** | GPS-tagged projects plotted on an interactive map. |
| **Account management** | Self-service password change, plus recent sign-in activity including failed attempts and originating IP. |

---

## Architecture

```
Browser
   │   HTML (Jinja2) + fetch() for validation and comparison
   ▼
Flask application  (app.py)
   │
   ├── auth.py         sessions, CSRF, role gating, login throttling
   ├── database.py     SQLite persistence, every query scoped by owner
   ├── vision.py       photo-to-photo change detection (OpenCV/NumPy)
   ├── seed_demo.py    demo account and sample project seeding
   │
   ├── TensorFlow/Keras   8 CNN models, lazy-loaded on first inference
   └── ReportLab          PDF audit report generation
   │
   ▼
SQLite  (data/crisp.db)
```

Three deliberate properties of this layout:

- **Nothing touches TensorFlow at import time.** The server accepts
  requests immediately; the roughly 5 second ensemble load happens on the
  first request that actually needs inference. Pages that only read stored
  results never trigger it.
- **Persistence is isolated behind `database.py`.** Routes never build SQL.
  Swapping SQLite for Postgres means rewriting one module.
- **Progress is computed, not stored.** Denormalised progress fields would
  drift out of sync with validation history. Deriving them on read makes
  that impossible.

---

## Construction model

Five weighted phases, each subdivided into weighted sub-stages. Both levels
sum to 100%.

| # | Stage | Weight | Sub-stages (weight within stage) |
|:-:|-------|:------:|----------------------------------|
| 1 | Foundation | 20% | Excavation (25), Reinforcement Placement (25), Concrete Curing (25), Concrete Pouring (25) |
| 2 | Superstructure | 30% | Structural Frame Erection (40), Structural Wall Construction (25), Stair Case (20), Roof Decking (15) |
| 3 | Facade | 20% | Exterior Wall Construction (40), Window &amp; Door Installation (35), Exterior Cladding &amp; Finishes (25) |
| 4 | Interior | 15% | Ceiling Installation (35), Flooring Installation (35), Staircase Finishing (30) |
| 5 | Finishing Works | 15% | Painting (35), Fixture Installation (35), Millwork &amp; Carpentry (30) |

**Overall completion** = sum of fully completed stage weights, plus the
current stage's weight scaled by its own sub-stage progress.

```
Project at Finishing Works / Painting

Foundation      100%  ->  20.0
Superstructure  100%  ->  30.0
Facade          100%  ->  20.0
Interior        100%  ->  15.0
Finishing        35%  ->  15.0 x 0.35 = 5.25
                         ─────────────────
                                     90.25  ->  90.3%
```

Sub-stages are cumulative within a stage: reaching sub-stage *n* implies
1..*n* are complete. Stages advance in order; a comparison that moves
backwards is reported as `invalid` rather than as a negative delta.

> **On that `90.25`.** Weighted integer percentages land on exact `.x5` ties
> constantly. Python's `%.1f` uses banker's rounding (to even) while
> JavaScript's `toFixed(1)` rounds half away from zero, so the same value
> rendered server-side and client-side disagreed: 90.2 against 90.3. It is
> now rounded once at source in `calculate_progress()` using round-half-up,
> so every consumer displays the same digit. A regression test pins it.

---

## Machine learning pipeline

### Stage 1: ensemble phase classification

Three ImageNet-pretrained backbones vote on which of the five phases an
image shows, using fixed weights:

| Model | Input | Weight |
|-------|:-----:|:------:|
| MobileNetV2 | 224x224 | 0.30 |
| InceptionV3 | 299x299 | 0.40 |
| VGG16 | 224x224 | 0.30 |

Softmax outputs are combined as a weighted sum. `argmax` gives the phase;
the value at that index is the confidence.

### Stage 2: sub-stage classification

The predicted phase selects a dedicated model, which classifies the
sub-stage within that phase only. Four use a MobileNetV2 backbone; Facade
uses InceptionV3 at 299x299.

This two-tier design is why sub-stage confidence runs far higher than phase
confidence. The second model separates 3 to 4 visually distinct activities
that already share a construction context, rather than discriminating
across the whole project lifecycle.

### Stage 3: description

Photograph descriptions come from a three-tier fallback:

1. **Any OpenAI-compatible vision gateway**, if `AI_GATEWAY_URL` is set.
   Works with a self-hosted runtime (Ollama, LocalAI, vLLM) or any hosted
   provider exposing an OpenAI-compatible `/v1` endpoint. The image is sent
   as a base64 data URL with its real MIME type.
2. **Google Gemini**, if `GOOGLE_API_KEY` is set.
3. **An offline generator** keyed on the classified stage and sub-stage.

Tier 3 is the default and needs no API key, no account and no network. The
application is fully functional with zero external services configured.
Each tier falls through to the next on timeout, HTTP error or malformed
response, so a misconfigured gateway degrades rather than breaking
validation. The tier actually in use is logged at startup.

Two practical notes. The configured model must accept image input; a
text-only model is rejected by the endpoint and CRISP falls back. And a
gateway bound to localhost is not reachable from a deployed container, so a
deployment either points at a hosted endpoint or uses the offline generator.
