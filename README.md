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

[Features](#features) · [Architecture](#architecture)
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
