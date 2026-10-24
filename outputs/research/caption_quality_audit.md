# Caption Quality Audit — Model-Generated Image Observations

**Audit date:** 2026-10-18 · **Store:** `outputs/annotation/model_generated/generation_records.csv`
**Model:** `Salesforce/blip-image-captioning-base` (local, CPU) · **Prompt:** `v1`
**Backend:** `local` · **Provenance:** every row `model_generated_observation`

This audit quantifies how informative the generated captions are. It is the evidence base
for the decision in §8 (which experimental track is scientifically defensible). All numbers
are measured from real artifacts; each is labelled with its measurement snapshot because
generation is still running and the store keeps growing.

## 1. Coverage at audit time

| Snapshot | Completed records | Unique descriptions | Eligible total |
|---|---:|---:|---:|
| 2026-10-18 11:25:44 | **7,863** | 2,259 | 17,870 |
| 2026-10-18 ~11:10 (`model_annotation_quality.json`) | 7,239 | — | 17,870 |

All completed records belong to the **train** split. Validation (1,751) and test (3,546)
have **zero** records, because the pipeline walks the manifest in order and the train split
is still being processed. Cross-split caption-quality comparison is therefore **not yet
possible** and is marked `NOT MEASURABLE YET`.

## 2. Structural checks (`outputs/research/model_annotation_quality.json`)

| Check | Count | Blocking | Meaning |
|---|---:|---|---|
| empty_or_malformed | 0 | no | no empty/too-short/boilerplate text |
| metadata_echo | 0 | no | no split/filename/metadata leakage |
| missing_or_invalid_provenance | 0 | yes | all rows carry valid provenance |
| invalid_model_metadata | 0 | yes | model/version/prompt hash present |
| image_identity_or_corruption | 0 | yes | image SHA-256 matches the manifest |
| duplicate_records | 0 | yes | no duplicate `sample_id` rows |
| class_label_leakage | **2** | no | text contains a PlantWild class name (`apple scab/google_0074.jpg`, `coffee leaf rust/google_0056.jpg`) |
| unsupported_disease_claim | **68** | no | asserts disease/cause/treatment terms (e.g. `disease`, `fungus`, `infect`) |
| missing_generations | 10,631 | no | store incomplete (expected; generation in progress) |

Overall verdict: **`EMITTABLE`** (`integrity_blocked: false`), 7,169 clean records,
70 records requiring human verification. Structural integrity is good; the problem is
**informativeness**, not malformation.

## 3. Informativeness metrics (snapshot 2026-10-18 11:25:44, n = 7,863)

| Measure | Value | Interpretation |
|---|---:|---|
| Unique descriptions | 2,259 / 7,863 (28.7%) | 71.3% of rows are exact repeats |
| Rows still containing the `v1` prompt echo | **7,863 (100%)** | every caption begins with the prompt text |
| Mean caption length | 15.46 words | ≈ 10 words are the fixed prefix |
| Largest identical-caption group | **1,070 rows** | one caption covers 1,070 different images |
| Duplicate-caption groups (size > 1) | 634 | |
| Duplicate groups spanning **> 1 disease class** | **453 (71.5%)** | identical text across different diseases |
| Rows sharing a caption with a **different class** | **5,674 (72.2%)** | |
| Top-5 captions | 1,070× "…the green leaves"; 202×/195× "…the damage of the …"; 179×/142× "…the brown spots on the …" | a handful of phrases dominate |

**The decisive finding:** 72.2% of captions are shared verbatim by images from **different
disease classes**, and a single caption represents 1,070 images. Text with this distribution
cannot separate classes — a text classifier trained on it is effectively near-constant, so any
apparent "fusion gain" would be an artifact, not evidence of symptom language.

## 4. Prompt-echo defect (implementation finding)

`ml/annotation/generation.py` strips the prefix only when
`text.casefold().startswith(prompt.text.casefold())`. BLIP decodes the prompt with a spaced
hyphen (`"a close - up view…"`) while `PROMPTS["v1"].text` uses `"a close-up view…"`, so the
comparison fails and **every** stored description keeps the full prompt echo. This is a real
defect: it inflates text length with non-informative prompt restatement. It was **not**
patch-changed mid-run, because altering text formatting partway through would make early and
late records inconsistent; it is recorded here as a defect to fix before any future corpus.

## 5. Prompt-variant probe (bounded test, n = 12 images, same images, CPU)

The mandate requires testing alternative prompts rather than assuming. Four prompt variants
were run on the same 12 already-captioned train images:

| Variant | Prompt | Unique / 12 | Mean words | Symptom-bearing |
|---|---|---:|---:|---:|
| `none` | (no prefix, plain captioning) | 12 | 9.3 | 2/12 |
| **`v1` (in production)** | "a close-up view of the plant surface showing" | **9** | **15.9** | **1/12** |
| `leaf_shows` | "the leaf surface shows" | 10 | 6.6 | 4/12 |
| `symptom_ask` | "symptoms visible on this leaf include" | **12** | 8.6 | **6/12** |

Findings:
- The **production `v1` prompt is the weakest of the four** on this sample (lowest uniqueness,
  longest text — mostly echo, fewest symptom words).
- A symptom-oriented prefix roughly **triples** symptom-bearing output (1/12 → 6/12) and
  restores full uniqueness.
- Absolute quality remains poor: the model misidentified subjects entirely on some images
  ("three different colored rocks on a white surface", "the different colors of the soil"),
  i.e. captions are frequently **factually wrong about the image**, not just generic.

Caveat: n = 12, single model, CPU, one run — directional, not statistical. It shows a heading
worth pursuing, not a validated fix.

## 6. Representative problematic cases

- `"a close - up view of the plant surface showing the fruit on the tree"` — 92 images spanning
  `apple black rot/*`, `apple rust/156.jpg` (cross-class collision).
- `"…showing the fruit and the fruit on the plant"` — spans `apple black rot`, `apple scab`.
- `"…showing the leaves and flowers"` — spans `apple black rot`, `apple leaf`, `apple mosaic virus`.
- `apple rust/google_0120.jpg` — asserts `fungus` (unsupported disease claim).
- `apple scab/google_0074.jpg` — caption contains a PlantWild class name (label leakage, excluded).

## 7. Split / class comparison

- **By split:** not measurable — only the train split has captions (validation and test are
  empty). This will be re-measured once generation reaches those splits.
- **By class (train):** measurable only as collision — 453 of 634 duplicate groups mix classes
  (§3). There is no evidence that captions distinguish classes.

## 8. Decision (per mandate §3)

The captions are **structurally valid but too generic, too repetitive, and too often wrong
about the subject** to support a meaningful symptom-language fusion result. 72.2% cross-class
caption sharing means the text cannot carry class-discriminative signal.

Chosen path: **C for the human-text experiment (blocked) + B only as an explicitly labelled
exploratory track once coverage exists.**

- **A (human text):** no genuinely human-authored per-image observations exist → the human-text
  fusion experiment stays **blocked** on that data.
- **B (model-generated exploratory):** permitted only as a clearly-labelled engineering
  experiment, **and** it is currently blocked anyway because validation has no captions, so the
  trainer's `train`+`validation` gate fails (`ValueError: Paired dataset needs at least one
  official train and validation example`). Any metrics produced by this track would **not**
  demonstrate benefit from genuine symptom descriptions.
- **C:** if neither A nor a defensible B is available, training is withheld and the exact
  dependency is reported.

> Generated captions are candidate machine text. They are **not** human observations, **not**
> expert-verified, and **not** independently reviewed. `model_generated_observation` is never
> relabelled as `user_observation`.

## 9. Recommended improvement (not yet implemented, pending a decision)

1. Fix the prefix-strip defect (§4) and adopt a symptom-oriented prompt (e.g. `symptom_ask`)
   as a **new prompt version** (`v3`) so the existing `v1` corpus stays immutable.
2. Evaluate the candidate prompt on a larger, class-balanced sample and quantify
   symptom-bearing rate and cross-class collision before committing to a re-generation.
3. Only then consider a `v3` corpus. A larger captioning model (e.g. BLIP-large or an
   instruction-following VLM) may be needed, but that is untested here and out of scope for
   the current CPU-only environment.
