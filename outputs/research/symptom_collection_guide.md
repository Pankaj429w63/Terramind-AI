# TerraMind AI — Symptom Observation Collection Guide

**Audience:** contributors ("simulated users") and the curator who assembles the paired dataset.
**Purpose:** describe what is *visibly present in one plant image* in plain language, so the text-only and fusion models can be trained on the same kind of text a real user would submit.
**Rule zero:** write only what you can see in the image. Never guess the cause.

## 1. Blinding — the most important rule

You will receive images as opaque aliases (`obs-xxxxxxxxxxxx`). You must not know or infer the answer.

- Work only from the image and its alias. You never see the file name, folder, path, class name, or whether the image is train/validation/test.
- Do **not** name the plant species or crop ("tomato", "apple", "cherry") unless the only thing you can honestly say is generic ("the leaf").
- Do **not** name a disease or a class from any list, and do **not** copy from any prompt, web page, or model output.
- Do **not** reverse-image-search, open image metadata, or look up the source of an image.
- Do **not** discuss an image with others during a session, and do not look at another contributor's rows.

If you recognise a disease, still describe only the visible features. Naming it is rejected by the validator and voids the row.

## 2. What counts as an observable symptom

Describe visible appearances, not diagnoses:

| Check | Describe |
|---|---|
| Organ | leaf, stem, fruit, whole plant — and which part is affected (upper/lower surface, edge, tip, base) |
| Marks | spots, blotches, streaks, lesions, ring patterns, specks |
| Colour | the colour(s) of the marks and of the surrounding tissue |
| Shape & size | round/oval/irregular/angular, tiny/pin-head/centimetre-scale (approximate) |
| Margin | sharp, diffuse, raised, sunken, with or without a halo |
| Surface | powdery, fuzzy, velvety, shiny, greasy, dry, cracked, oozing |
| Distribution | scattered, clustered, along veins, in rings, concentrated at the edge |
| Extent | a few marks, about a quarter, most of the surface |
| Other signs | wilting, curling, yellowing, holes, webbing, insects or frass if visible |

## 3. How to phrase it

Write **1–3 short sentences**, plain language, no jargon and no Latin names. Use frames like these, completed with what you actually see:

- "The `<organ>` has `<number/extent>` `<colour>` `<shape>` marks, about `<size>`, with `<margin>`, mostly on `<location>`."
- "A `<colour>` `<powdery/fuzzy/velvety>` coating covers `<extent>` of the `<surface>`."
- "The `<organ>` is `<wilting/curling/discoloured>` and the `<edge/veins/tip>` show `<feature>`."

**These are sentence frames, not example data.** Never paste a completed sentence from this guide, from another row, or from any other image. If you are unsure, say so in the sentence ("possible early lesion, low confidence") instead of leaving it blank or copying.

## 4. What gets rejected, and why

| You wrote | Result | Reason |
|---|---|---|
| A disease or class name (e.g. a full "species + disease" phrase) | **Rejected** | Target leakage — the text must not encode the label |
| Any recognised PlantWild class phrase | **Rejected** | Same rule, enforced by `validate_text_not_label_derived` |
| Blank observation | Skipped, not emitted | Treated as "not yet annotated"; never auto-filled |
| Observation with no `observer_id` | **Rejected** | The observation must be attributable to a real person |
| Identical sentence pasted for many images | **Rejected at review** | Not a genuine image-specific observation |
| Extra columns such as `image_path`, `class_name`, `split` in the workbook | **Rejected** | Breaks blinding |

## 5. Session procedure

1. The curator runs `python -m scripts.build_annotation_workbook` and hands you only the workbook.
2. You are shown each image one alias at a time; view it without any filename or metadata.
3. For each alias, fill exactly one row:
   - `alias` — as given (do not edit).
   - `observation` — your 1–3 sentence description.
   - `observer_id` — your pseudonymous identifier.
   - `observed_at` — the date you made the observation, ISO 8601 (`YYYY-MM-DD`).
4. Do not reorder, rename, or add columns. One row per alias.
5. Submit the workbook to the curator. Do not mark anything as approved yourself.
6. The curator compiles the review ledger and runs the validator.

## 6. Independence, review, and disagreement

- A held-out subset (about 5%) is described independently by a second contributor who cannot see the first description. This measures agreement on the visible features (organ, mark presence, colour, distribution).
- A third, independent reviewer adjudicates disagreements using only the image. The adjudicated text is one of the two submissions, or a faithful summary of visible features — never new invented content.
- Rows that remain `disputed` are excluded. Only rows marked `approved` by a reviewer are used.
- If agreement on the held-out subset is poor, the guide is revised and re-piloted before the full run.

## 7. Why this matters

The production app asks the submitting user for a short description alongside the photo. Training text and inference text must come from the same instrument, or the comparison is invalid. Your descriptions are the training side of that instrument — describe the image as a user would, describing only what is visible.

## 8. Never do this

- Never invent, translate, summarise, or auto-generate a description.
- Never reuse a description from another image or another contributor.
- Never name a disease, class, or (where avoidable) the plant species.
- Never record an observation you did not personally make.
- Never mark your own rows `approved`.
