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
[ML pipeline](#machine-learning-pipeline) ·
[Change detection](#change-detection-pipeline) ·
[Metrics](#technical-metrics)
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

---

## Change detection pipeline

Comparing two site photographs taken days or weeks apart is not a pixel
subtraction problem. The naive version of this feature highlighted the
entire frame whenever the weather differed between visits, because a sunny
photograph and an overcast one differ in nearly every pixel.

The current pipeline (`vision.py`):

| # | Step | Purpose |
|:-:|------|---------|
| 1 | ORB features + RANSAC homography | Aligns the photos so drone repositioning is not read as change |
| 2 | Linear gain/offset matching, then CLAHE | Removes exposure and white-balance differences between visits |
| 3 | Local SSIM as `(1 - SSIM) / 2` | Compares *structure*, not brightness. A wall that merely got brighter scores near zero; a wall that appeared scores high |
| 4 | Construction-region masking | Restricts results to built structure, excluding sky, vegetation and smooth ground |
| 5 | Otsu threshold with an absolute floor | Adapts to the image while preventing noise being promoted to "change" on an unchanged site |
| 6 | Morphological open/close, hole fill, component-area filter | Produces contiguous regions rather than speckle |
| 7 | Tinted fill plus contour outline | Shows what changed, and where its boundary is |

### Separating sky from concrete

The hardest part is that pale concrete and white cladding are bright and
desaturated, exactly like an overcast sky. A brightness-only rule swallows
the buildings the feature exists to analyse. During development a
brightness-and-connectivity mask consumed **74%** of a newly-added
structure.

Sky is therefore identified by three properties *together*:

- Bright and desaturated, **or** blue-hued
- **Textureless**, meaning local edge density below threshold. Buildings carry
  edges, openings and panel lines even when flat-toned; sky does not
- **Top-anchored**, with the component's centroid in the upper third. A
  region that merely grazes the top edge but extends far down is a building

Adding the texture test took detection of the new structure from 31% to
**100%**.

### Optional semantic segmentation

If `models/segformer.onnx` is present, the region mask comes from SegFormer
semantic segmentation instead of the heuristic. Preprocessing is
implemented in NumPy, so this path needs `onnxruntime` alone: no PyTorch,
no `transformers`. The application reports which method actually ran on
every comparison rather than claiming a capability it did not use.

---

## Technical metrics

All figures measured on the reference machine: Windows 11, Python 3.11,
CPU-only inference, no GPU. Representative, not competitive benchmarks.

### Headline

| Measurement | Value |
|-------------|-------|
| **Full validation** (4 forward passes, decode to result) | **211 ms** mean · p50 218 · p95 235 · min 167 |
| Ensemble inference, 3 models | 173 ms mean · p95 192 |
| Sub-stage inference, 1 model | 31 ms mean · p95 44 |
| **Change detection**, aligned pair at 1400 px | **154 ms** mean · p95 164 |
| **PDF audit report** with embedded photograph | **99 ms** mean · p95 118 · 326 KB out |
| Authenticated page render, SQLite-backed | 2 to 8 ms |
| Resident memory, typical session | ~841 MB |
| Test suite | 44 tests in 5.5 s |

### Latency, in full

<details>
<summary><strong>Inference and model loading</strong></summary>

<br>

| Measurement | Value |
|-------------|-------|
| Import `app.py` | 0.20 s |
| Load the 3 ensemble models (first inference request) | 5.0 s |
| Load one stage-specific model (first use of that stage) | 0.74 s to 1.77 s |
| Ensemble inference, 3 models | 173 ms mean (p95 192 ms) |
| Sub-stage inference, 1 model | 31 ms mean (p95 44 ms) |
| **Full validation, 4 forward passes** | **211 ms** mean (p50 218, p95 235, min 167) |
| JPEG decode, 12 MP photo, reduced-resolution decode | 17 ms |
| JPEG decode, 12 MP photo, full resolution | 73 ms |
| Server ready to serve requests after launch | under 1 s |
| Clean `pip install -r requirements.txt` | 7 min 11 s (~70 packages) |

</details>

<details>
<summary><strong>Change detection runtime</strong></summary>

<br>

Measured on a 1280x720 source photograph; working resolution is capped at
1400 px wide.

| Scenario | Mean | p95 |
|----------|-----:|----:|
| Aligned pair, lighting drift only | 154 ms | 164 ms |
| Aligned pair, lighting drift plus a new structure | 185 ms | 197 ms |
| Non-overlapping viewpoints (ORB cannot converge) | 251 ms | 274 ms |

The worst case is the one where alignment fails, because RANSAC exhausts
its iteration budget before the pipeline falls back to an unaligned
comparison. It is still under 300 ms, and the interface reports that
alignment did not succeed rather than presenting a confident-looking
result.

</details>

<details>
<summary><strong>Web and database layer</strong></summary>

<br>

Flask test client against a warm SQLite database holding one user, one
project and 30 validations.

| Endpoint | Mean | p95 |
|----------|-----:|----:|
| `/home?project_id=...` project workspace | 8.0 ms | 11.6 ms |
| `/expert_dashboard` | 6.7 ms | 9.2 ms |
| `/geo-map` | 3.4 ms | 5.3 ms |
| `/account` password form and sign-in history | 2.1 ms | 3.7 ms |
| `/healthz` | 0.3 ms | 0.4 ms |

| Database operation | Mean | p95 |
|--------------------|-----:|----:|
| `list_projects(user_id)` | 2.0 ms | 2.7 ms |
| `list_validations(user_id, project_id)`, 30 rows | 2.0 ms | 2.6 ms |
| `get_latest_validation(user_id, project_id)` | 1.7 ms | 2.7 ms |
| `verify_user()`, scrypt `32768:8:1` | 94 ms | 120 ms |

The 94 ms credential check is the **intended** cost, not a regression:
scrypt is configured to be expensive so offline cracking of a stolen hash
is expensive. It is paid once per sign-in. Login throttling caps an
attacker at 8 attempts per IP and email pair.

| Report generation | Mean | Output |
|-------------------|-----:|-------:|
| PDF with embedded photograph | 99 ms | 326 KB |
| PDF, text only | 22 ms | 2 KB |

Database file size: **76 KB** for one user, one project and 30
validations, including the audit log.

</details>

### Why the inference path is written the way it is

For a single sample the obvious implementation is the slowest. Running the
full three-model ensemble once, same inputs, same machine:

| Strategy | Mean | Relative |
|----------|-----:|---------:|
| `model.predict()` | 359 ms | 1.00x |
| `model(x, training=False)` in eager mode | 868 ms | 2.42x slower |
| `model.predict_on_batch()` | 162 ms | 0.45x |
| **`tf.function`-traced call** | **155 ms** | **0.43x** |

`.predict()` builds a `tf.data` pipeline on every call, which dominates the
forward pass at batch size 1. Calling the model directly avoids that but
runs op-by-op in eager mode, which is worse again. Tracing the call once
into a graph and reusing it is 2.3x faster than `.predict()` and produces
bit-identical output: maximum absolute difference 0.0.

Two further wins, both measured:

- **Stage models load on demand.** Only one of the five sub-stage models is
  needed per validation. Loading all five up front cost about 285 MB and
  several seconds for models most sessions never call. The trade: a session
  touching all five ends slightly *higher* (~1,036 MB) than eager loading
  did (~950 MB), because each compiled graph carries its own overhead. The
  common case improved; the worst case is marginally worse.
- **Large uploads decode at reduced resolution.** The largest model input
  is 299x299, so fully decoding a 12 megapixel photograph is wasted work.
  Letting the JPEG decoder downscale during decode takes that step from
  73 ms to 17 ms, and makes no difference to already-small images.

### Memory

| State | Resident |
|-------|---------:|
| Process baseline, no models | ~55 MB |
| Ensemble loaded (3 models) | ~800 MB |
| Ensemble + 1 stage model (typical session) | ~841 MB |
| Ensemble + all 5 stage models | ~1,036 MB |

The working set, roughly 841 MB typically and 1,036 MB if every stage is
exercised, is the single most important deployment number. It rules out
512 MB free tiers, and it is why the container runs **one Gunicorn
worker**: every additional worker loads its own complete copy of the
models, multiplying memory without adding throughput.

### Model weights

Distributed as GitHub Release assets, not committed to the repository. Four
files exceed GitHub's 100 MB per-file limit, and Git LFS on a free account
permits roughly one clone per month before the bandwidth quota is exhausted.

| File | Size | Role |
|------|-----:|------|
| `vgg16.keras` | 184.5 MB | Ensemble member (0.30) |
| `inception.keras` | 151.0 MB | Ensemble member (0.40) |
| `Facade_inception.keras` | 151.0 MB | Facade sub-stages |
| `mobilenet.keras` | 33.8 MB | Ensemble member (0.30) |
| `Foundation_mobile.keras` | 33.8 MB | Foundation sub-stages |
| `Superstructure_mobile.keras` | 33.8 MB | Superstructure sub-stages |
| `Interior_mobile.keras` | 33.8 MB | Interior sub-stages |
| `finishing_mobile.keras` | 33.8 MB | Finishing sub-stages |
| **Total** | **655.5 MB** | 8 files |

The three ensemble models (369 MB) load on the first request needing
inference. The five stage-specific models load individually, on first use
of that stage.

### Classification confidence on the bundled samples

Measured against the five photographs in `static/demo_samples/`. Each is a
real construction photograph that the ensemble classifies into its own
declared stage.

| Sample | Phase confidence | Sub-stage | Sub-stage confidence |
|--------|-----------------:|-----------|---------------------:|
| Foundation | 87.1% | Excavation | 100.0% |
| Interior | 74.0% | Ceiling Installation | 99.9% |
| Facade | 73.1% | Exterior Wall Construction | 100.0% |
| Superstructure | 72.7% | Structural Frame Erection | 97.3% |
| Finishing Works | 60.4% | Fixture Installation | 99.4% |

> **Read these correctly.** They are inference outputs on five
> photographs, not an evaluation. No held-out labelled set ships with this
> project, so there are no accuracy, precision, recall or confusion-matrix
> figures. The 60 to 87% phase against 97 to 100% sub-stage gap is structural,
> explained under [stage 2](#stage-2-sub-stage-classification).

### Change detection accuracy

Measured on a controlled reproduction of the real failure mode: the same
photograph dimmed to 72% brightness, lifted 28 levels for haze,
blue-shifted, blurred, and offset by 9 pixels, simulating a sunny-to-overcast revisit
with drone drift.

| Scenario | Result |
|----------|--------|
| Lighting change only, no structural change | **0.1%** of built area reported as changed |
| Identical photographs | 0.0%, reported as "no change" |
| Lighting change **plus** a new structure, recall | **100%** of the new structure detected |
| Same test, spill outside the new structure | 1.9% of the remaining frame |
| Same test, precision | 82.8% |

The first row is the headline: an entire frame of illumination change now
produces essentially zero reported progress.

### Codebase

| Component | Lines |
|-----------|------:|
| `app.py` (routes, inference, progress) | 1,577 |
| `database.py` (persistence, throttling, audit) | 535 |
| `test_app.py` (test suite) | 495 |
| `vision.py` (change detection) | 441 |
| `auth.py` (sessions, CSRF, role gating) | 193 |
| `seed_demo.py` | 161 |
| `download_models.py` | 146 |
| `init_db.py` | 68 |
| `build_models.py` | 62 |
| **Python total** | **~3,678** |
| Jinja2 templates (10 files) | 1,553 |
| CSS + JavaScript | 848 |
| **Total** | **~6,079** |
