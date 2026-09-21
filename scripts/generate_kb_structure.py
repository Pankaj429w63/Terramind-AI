"""
Generate TerraMind AI Agricultural Knowledge Base Structure.
Creates folder hierarchy, JSON schemas, class mappings, and empty template files.
DOES NOT invent agricultural facts — all document bodies contain placeholders only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_ROOT = PROJECT_ROOT / "data" / "knowledge_base"

CATEGORIES = ["diseases", "treatment", "fertilizer", "plant_care", "agriculture"]
RAG_CATEGORY_MAP = {
    "diseases": "diseases",
    "treatment": "treatment",
    "fertilizer": "fertilizer",
    "plant_care": "plant care",
    "agriculture": "agriculture",
}

RELIABILITY_VALUES = ["peer_reviewed", "government", "extension", "industry", "community", "unverified"]
REGION_EXAMPLES = ["global", "north_america", "europe", "asia", "africa", "south_america", "oceania"]

CLASSES_RAW = """0 apple black rot
1 apple leaf
2 apple mosaic virus
3 apple rust
4 apple scab
5 banana leaf
6 banana panama disease
7 basil downy mildew
8 basil leaf
9 bean halo blight
10 bean leaf
11 bean mosaic virus
12 bean rust
13 bell pepper leaf
14 bell pepper leaf spot
15 blueberry leaf
16 blueberry rust
17 broccoli downy mildew
18 broccoli leaf
19 cabbage alternaria leaf spot
20 cabbage leaf
21 carrot cavity spot
22 cauliflower alternaria leaf spot
23 cauliflower leaf
24 celery anthracnose
25 celery early blight
26 celery leaf
27 cherry leaf
28 cherry leaf spot
29 cherry powdery mildew
30 citrus canker
31 citrus greening disease
32 coffee leaf
33 coffee leaf rust
34 corn gray leaf spot
35 corn leaf
36 corn northern leaf blight
37 corn rust
38 corn smut
39 cucumber angular leaf spot
40 cucumber bacterial wilt
41 cucumber leaf
42 cucumber powdery mildew
43 eggplant cercospora leaf spot
44 eggplant leaf
45 garlic leaf
46 garlic leaf blight
47 garlic rust
48 ginger leaf
49 ginger leaf spot
50 ginger sheath blight
51 grape black rot
52 grape downy mildew
53 grape leaf
54 grape leaf spot
55 grapevine leafroll disease
56 lettuce downy mildew
57 lettuce leaf
58 lettuce mosaic virus
59 maple leaf
60 maple tar spot
61 peach leaf
62 peach leaf curl
63 plum leaf
64 plum pocket disease
65 potato early blight
66 potato late blight
67 potato leaf
68 raspberry leaf
69 rice blast
70 rice leaf
71 rice sheath blight
72 soybean leaf
73 squash leaf
74 squash powdery mildew
75 strawberry anthracnose
76 strawberry leaf
77 strawberry leaf scorch
78 tobacco leaf
79 tobacco mosaic virus
80 tomato bacterial leaf spot
81 tomato early blight
82 tomato late blight
83 tomato leaf
84 tomato leaf mold
85 tomato mosaic virus
86 tomato septoria leaf spot
87 tomato yellow leaf curl virus
88 zucchini yellow mosaic virus"""


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def parse_classes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in CLASSES_RAW.splitlines():
        if not line.strip():
            continue
        idx_str, label = line.strip().split(maxsplit=1)
        idx = int(idx_str)
        tokens = label.split()
        is_healthy = tokens[-1] == "leaf" and len(tokens) <= 2
        if is_healthy:
            crop = " ".join(tokens[:-1]).strip() or tokens[0]
            disease = None
            class_type = "healthy_leaf"
        else:
            if tokens[-1] in {"leaf"} and len(tokens) > 2:
                crop = " ".join(tokens[:-1])
                disease = None
                class_type = "healthy_leaf"
            else:
                crop_tokens: list[str] = []
                disease_tokens: list[str] = []
                crop_keywords = {"apple", "banana", "basil", "bean", "bell", "blueberry",
                                 "broccoli", "cabbage", "carrot", "cauliflower", "celery",
                                 "cherry", "citrus", "coffee", "corn", "cucumber", "eggplant",
                                 "garlic", "ginger", "grape", "grapevine", "lettuce", "maple",
                                 "peach", "plum", "potato", "raspberry", "rice", "soybean",
                                 "squash", "strawberry", "tobacco", "tomato", "zucchini"}
                hit_crop = False
                for t in tokens:
                    if t.lower() in crop_keywords or (hit_crop and t.lower() == "pepper"):
                        crop_tokens.append(t)
                        if t.lower() in {"bell"}:
                            pass
                        else:
                            hit_crop = True
                    else:
                        if hit_crop:
                            disease_tokens.append(t)
                        else:
                            crop_tokens.append(t)
                crop = " ".join(crop_tokens).strip()
                disease = " ".join(disease_tokens).strip() or None
                class_type = "disease" if disease else "healthy_leaf"
                if not disease and class_type == "healthy_leaf":
                    pass
        if not disease and class_type != "healthy_leaf":
            class_type = "healthy_leaf"
        if not disease:
            crop = label.replace(" leaf", "").strip()
            if crop == label:
                parts = label.split()
                if len(parts) == 2 and parts[1] == "leaf":
                    crop = parts[0]
        rows.append({
            "class_index": idx,
            "class_label": label,
            "class_slug": slug(label),
            "crop": crop,
            "crop_slug": slug(crop),
            "disease": disease,
            "disease_slug": slug(disease) if disease else None,
            "class_type": class_type,
        })
    return rows


DOCUMENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "terramind://schemas/knowledge-document-v1.schema.json",
    "title": "TerraMind Knowledge Document",
    "description": "Structured agricultural knowledge document for RAG ingestion.",
    "type": "object",
    "required": ["schema_version", "document_id", "category", "source", "title", "crop",
                 "content", "url", "author", "date", "region", "reliability", "metadata"],
    "properties": {
        "schema_version": {"type": "string", "const": "1.0.0"},
        "document_id": {
            "type": "string",
            "description": "Deterministic document UUID v5 in terramind:kb namespace (category:crop_slug:disease_slug?)"
        },
        "category": {
            "type": "string",
            "enum": ["diseases", "treatment", "fertilizer", "plant care", "agriculture"]
        },
        "source": {
            "type": "string",
            "description": "Human-readable source identifier, e.g. USDA Extension, FAO, CABI, peer journal name"
        },
        "title": {
            "type": "string",
            "description": "Document title for retrieval display"
        },
        "crop": {
            "type": ["string", "null"],
            "description": "Crop common name in English (matched to PlantWild crop token)"
        },
        "disease": {
            "type": ["string", "null"],
            "description": "Disease / condition common name in English; null for agriculture/plant care docs"
        },
        "content": {
            "type": "string",
            "minLength": 1,
            "description": "Plain-text factual agricultural content (NOT fabricated — placeholder until sourced)."
        },
        "url": {
            "type": "string",
            "format": "uri",
            "description": "Publicly accessible URL of the authoritative source; placeholder until sourced."
        },
        "author": {
            "type": "string",
            "description": "Author, institution or issuing body name"
        },
        "date": {
            "type": "string",
            "format": "date",
            "description": "Publication date in ISO 8601 YYYY-MM-DD"
        },
        "region": {
            "type": "string",
            "enum": REGION_EXAMPLES,
            "description": "Primary geographic applicability region"
        },
        "reliability": {
            "type": "string",
            "enum": RELIABILITY_VALUES,
            "description": "Source trust tier (higher enumeration = more authoritative)"
        },
        "language": {
            "type": "string",
            "default": "en"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "metadata": {
            "type": "object",
            "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
            "description": "Arbitrary key/value payload passed through to the RAG vector store (Document.metadata)."
        }
    }
}


def empty_document(category: str, crop: str, disease: str | None,
                   class_label: str) -> dict[str, Any]:
    import uuid
    ns = uuid.uuid5(uuid.NAMESPACE_URL, "https://terramind.ai/schemas/kb")
    key = f"{category}:{slug(crop)}:{slug(disease) if disease else 'none'}"
    doc_id = str(uuid.uuid5(ns, key))
    placeholders = {
        "diseases": {
            "title": f"[PLACEHOLDER] {class_label} — Disease Profile",
            "source": "[SOURCE REQUIRED: e.g. USDA APHIS, CABI Compendium, extension bulletin]",
            "url": "https://example.com/REPLACE-WITH-AUTHORITATIVE-SOURCE",
            "author": "[Institution/Author to be provided]",
            "content": (
                "[REQUIRED: Factual disease description to be sourced from authoritative "
                "agricultural extension / peer-reviewed literature / IPM databases. Must "
                "include causal organism, hosts, symptoms (leaf/fruit/stem), disease cycle, "
                "favorable environmental conditions, spread mechanisms, diagnostic features."
                " Do not fabricate — leave this note in place until real content is inserted.]"
            ),
        },
        "treatment": {
            "title": f"[PLACEHOLDER] {class_label} — Treatment & IPM",
            "source": "[SOURCE REQUIRED: e.g. land-grant extension, manufacturer label, IPM guide]",
            "url": "https://example.com/REPLACE-WITH-AUTHORITATIVE-SOURCE",
            "author": "[Institution/Author to be provided]",
            "content": (
                "[REQUIRED: Cultural / biological / chemical treatment options, application "
                "timing, resistance management, harvest intervals, organic vs conventional "
                "options, regulatory restrictions (region-specific). Source from label "
                "documents, extension recommendations, and IPM handbooks only.]"
            ),
        },
        "fertilizer": {
            "title": f"[PLACEHOLDER] {crop} — Fertilizer & Nutrition Guide",
            "source": "[SOURCE REQUIRED: e.g. FAO fertilizer manual, soil extension service]",
            "url": "https://example.com/REPLACE-WITH-AUTHORITATIVE-SOURCE",
            "author": "[Institution/Author to be provided]",
            "content": (
                "[REQUIRED: Macronutrient (N/P/K) and micronutrient requirements by growth "
                "stage, deficiency/toxicity symptoms tied to {crop}, soil pH targets, "
                "application rates and timing, fertigation guidance, organic amendments.]"
            ).format(crop=crop),
        },
        "plant care": {
            "title": f"[PLACEHOLDER] {crop} — Plant Care & Crop Management",
            "source": "[SOURCE REQUIRED: e.g. extension crop production guide]",
            "url": "https://example.com/REPLACE-WITH-AUTHORITATIVE-SOURCE",
            "author": "[Institution/Author to be provided]",
            "content": (
                "[REQUIRED: Planting dates, spacing, pruning/trellising, irrigation "
                "scheduling, pollination needs, scouting cadence, harvesting maturity "
                "indicators, storage post-harvest handling for {crop}.]"
            ).format(crop=crop),
        },
        "agriculture": {
            "title": f"[PLACEHOLDER] {crop} — Agricultural Production Overview",
            "source": "[SOURCE REQUIRED: e.g. FAO, USDA ERS, national ag statistics]",
            "url": "https://example.com/REPLACE-WITH-AUTHORITATIVE-SOURCE",
            "author": "[Institution/Author to be provided]",
            "content": (
                "[REQUIRED: Agro-ecological zones suitable for {crop}, typical yield "
                "ranges, crop rotation partners, cover-cropping integration, economic "
                "importance, major production regions, climate adaptation strategies.]"
            ).format(crop=crop),
        },
    }
    ph = placeholders[category]
    return {
        "schema_version": "1.0.0",
        "document_id": doc_id,
        "category": category,
        "source": ph["source"],
        "title": ph["title"],
        "crop": crop,
        "disease": disease,
        "content": ph["content"],
        "url": ph["url"],
        "author": ph["author"],
        "date": "0000-00-00",
        "region": "global",
        "reliability": "unverified",
        "language": "en",
        "tags": ["placeholder", "needs-source", slug(crop)],
        "metadata": {
            "class_label": class_label,
            "crop_slug": slug(crop),
            "disease_slug": slug(disease) if disease else "",
            "status": "placeholder",
            "placeholder_replaced": "false",
            "last_reviewed": "",
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    classes = parse_classes()
    KB_ROOT.mkdir(parents=True, exist_ok=True)

    schemas_dir = KB_ROOT / "_schemas"
    templates_dir = KB_ROOT / "_templates"

    write_json(schemas_dir / "knowledge-document-v1.schema.json", DOCUMENT_SCHEMA)

    mapping: dict[str, Any] = {
        "schema_version": "1.0.0",
        "total_classes": len(classes),
        "categories": CATEGORIES,
        "rag_category_aliases": RAG_CATEGORY_MAP,
        "reliability_values": RELIABILITY_VALUES,
        "region_values": REGION_EXAMPLES,
        "classes": [],
        "crops": [],
    }

    crop_set: dict[str, dict[str, Any]] = {}

    for c in classes:
        categories_needed: list[str] = []
        if c["class_type"] == "disease":
            categories_needed.extend(["diseases", "treatment", "fertilizer", "plant_care", "agriculture"])
        else:
            categories_needed.extend(["fertilizer", "plant_care", "agriculture"])

        rag_cats = [RAG_CATEGORY_MAP[c_] for c_ in categories_needed]

        mapping["classes"].append({
            "class_index": c["class_index"],
            "class_label": c["class_label"],
            "class_slug": c["class_slug"],
            "crop": c["crop"],
            "crop_slug": c["crop_slug"],
            "disease": c["disease"],
            "disease_slug": c["disease_slug"],
            "class_type": c["class_type"],
            "kb_categories": categories_needed,
            "rag_categories": rag_cats,
            "document_count_expected": len(categories_needed),
        })

        cs = c["crop_slug"]
        if cs not in crop_set:
            crop_set[cs] = {
                "crop": c["crop"],
                "crop_slug": cs,
                "classes": [],
                "disease_classes": 0,
                "healthy_classes": 0,
                "categories_expected": ["fertilizer", "plant_care", "agriculture"],
            }
        crop_set[cs]["classes"].append({
            "index": c["class_index"],
            "label": c["class_label"],
            "type": c["class_type"],
            "disease": c["disease"],
        })
        if c["class_type"] == "disease":
            crop_set[cs]["disease_classes"] += 1
        else:
            crop_set[cs]["healthy_classes"] += 1
        if c["class_type"] == "disease" and "diseases" not in crop_set[cs]["categories_expected"]:
            crop_set[cs]["categories_expected"].extend(["diseases", "treatment"])

    mapping["crops"] = sorted(crop_set.values(), key=lambda x: x["crop_slug"])

    write_json(schemas_dir / "class_knowledge_mapping.json", mapping)

    for cat, rag_cat in RAG_CATEGORY_MAP.items():
        template = empty_document(rag_cat, "CROP_NAME", None if cat in {"fertilizer", "plant_care", "agriculture"} else "DISEASE_NAME",
                                  "CLASS_LABEL")
        write_json(templates_dir / f"{cat}.template.json", template)

    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "kb_root": str(KB_ROOT),
        "total_classes": len(classes),
        "total_crops": len(crop_set),
        "categories": CATEGORIES,
        "rag_category_aliases": RAG_CATEGORY_MAP,
        "category_manifests": {},
    }

    for cat in CATEGORIES:
        rag_cat = RAG_CATEGORY_MAP[cat]
        cat_dir = KB_ROOT / cat
        cat_manifest: dict[str, Any] = {
            "category": cat,
            "rag_category": rag_cat,
            "total_documents": 0,
            "crops": {},
        }
        for crop_slug, info in crop_set.items():
            if cat in {"diseases", "treatment"} and info["disease_classes"] == 0:
                continue
            crop_dir = cat_dir / info["crop_slug"]
            crop_dir.mkdir(parents=True, exist_ok=True)
            docs_in_crop = 0
            crop_manifest_list: list[dict[str, Any]] = []
            if cat in {"diseases", "treatment"}:
                for cls in info["classes"]:
                    if cls["type"] != "disease":
                        continue
                    fn = f"{slug(cls['label'])}.{cat}.json"
                    fpath = crop_dir / fn
                    doc = empty_document(rag_cat, info["crop"], cls["disease"], cls["label"])
                    write_json(fpath, doc)
                    crop_manifest_list.append({
                        "file": fn,
                        "document_id": doc["document_id"],
                        "class_index": cls["index"],
                        "class_label": cls["label"],
                        "disease": cls["disease"],
                        "status": doc["metadata"]["status"],
                    })
                    docs_in_crop += 1
            else:
                fn = f"{info['crop_slug']}.{cat}.json"
                fpath = crop_dir / fn
                doc = empty_document(rag_cat, info["crop"], None, info["crop"])
                write_json(fpath, doc)
                crop_manifest_list.append({
                    "file": fn,
                    "document_id": doc["document_id"],
                    "crop": info["crop"],
                    "status": doc["metadata"]["status"],
                })
                docs_in_crop += 1

            if docs_in_crop:
                cat_manifest["crops"][info["crop_slug"]] = {
                    "crop": info["crop"],
                    "document_count": docs_in_crop,
                    "documents": crop_manifest_list,
                }
                cat_manifest["total_documents"] += docs_in_crop

        write_json(cat_dir / f"_manifest.{cat}.json", cat_manifest)
        manifest["category_manifests"][cat] = {
            "path": f"{cat}/_manifest.{cat}.json",
            "rag_category": rag_cat,
            "document_count": cat_manifest["total_documents"],
        }

    total_docs = sum(v["document_count"] for v in manifest["category_manifests"].values())
    manifest["total_documents_expected"] = total_docs
    write_json(KB_ROOT / "knowledge_base.manifest.json", manifest)

    ingestion_manifest: dict[str, Any] = {
        "version": "1.0.0",
        "description": "Ingestion source list consumable by KnowledgePipeline.ingest([]).",
        "sources": [],
    }
    for cat in CATEGORIES:
        cat_dir = KB_ROOT / cat
        if not cat_dir.is_dir():
            continue
        ingestion_manifest["sources"].append({
            "source": str(cat_dir.resolve()),
            "category": RAG_CATEGORY_MAP[cat],
            "format": "folder_of_json",
            "loader_note": "Each JSON file follows _schemas/knowledge-document-v1.schema.json; "
                           "the `content` field must be concatenated/embedded into the Document.text "
                           "field together with title/crop/disease for retrieval.",
        })
    write_json(KB_ROOT / "rag_ingestion.manifest.json", ingestion_manifest)

    print("Created TerraMind Knowledge Base at", KB_ROOT)
    print(f"   Classes: {len(classes)}  |  Crops: {len(crop_set)}  |  Documents: {total_docs}")
    print(f"   Categories: {CATEGORIES}")


if __name__ == "__main__":
    main()
