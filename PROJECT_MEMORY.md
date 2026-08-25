# TerraMind AI — Project Memory

> IMPORTANT FOR AI:
> Before making any change, read this file completely.
> This file is the permanent memory of the TerraMind AI project.
> If the user returns after days or weeks, first read this file, then read CURRENT_STATUS.md, TODO.md, DECISIONS.md and the latest SESSION_LOG.md.
> Never assume that undocumented work has been completed.

---

# 1. Project Identity

## Project Name

**TerraMind AI: A Multimodal Intelligent Agricultural Decision Support System**

## Project Type

Final-Year B.Tech AIML Project

## Domain

- When asked to train the model, run local CPU-safe training first and fall back to Docker or remote.
- Use `scripts/validate_dataset.py` to validate datasets before training; it writes `outputs/metrics/dataset_report.json`.
Agriculture

## SDG Alignment

**SDG 2 — Zero Hunger**

## Core Idea
- Agricultural knowledge
- Tables and structured data
- Weather and contextual information
- Historical crop information

The system will provide:

- Plant disease detection
- AI-assisted diagnosis
- Confidence scores
- Top disease predictions
- Evidence-based agricultural recommendations
- RAG-based agricultural knowledge retrieval
- Conversational AI
- AI agents
- Multi-agent orchestration
- Weather-aware context
- Farm and crop history
- Human/expert review
- AI-generated reports
- Full-stack web application
- SaaS architecture

---

# 2. Main Problem

Farmers often struggle to make timely and informed agricultural decisions because agricultural information is scattered across research papers, government documents, agricultural experts and different data sources.

Existing solutions often focus on only one modality, such as image-based disease detection.

TerraMind AI will integrate multimodal agricultural information into one intelligent decision support platform.

---

# 3. Current Project Goal

The project is being developed progressively.

The development order is:

1. Research and MVPDR baseline
2. Dataset preparation
3. Model training and evaluation
4. Production ML inference
5. FastAPI backend
6. Next.js frontend
7. Database
8. RAG
9. Local LLM
10. AI agent
11. Multi-agent system
12. Human-in-the-loop expert review
13. Reports
14. SaaS features
15. Evaluation
16. MLOps
17. Testing
18. Security
19. Monitoring
20. Docker
21. CI/CD
22. Deployment

DO NOT attempt to build all systems simultaneously.

---

# 4. Research Foundation

## Primary Research Model

**MVPDR**

Original repository:

https://github.com/tqwei05/MVPDR

Research paper:

https://doi.org/10.1145/3664647.3680599

## Primary Dataset

**PlantWild**

Official website:

https://tqwei05.github.io/PlantWild/

Dataset:

https://huggingface.co/datasets/uqtwei2/PlantWild

## Secondary Datasets

### PlantVillage

https://github.com/spMohanty/PlantVillage-Dataset

### PlantDoc

https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset

## Vision Foundation

OpenAI CLIP:

https://github.com/openai/CLIP

Tip-Adapter:

https://github.com/gaopengcuhk/Tip-Adapter

---

# 5. Technology Stack

## Machine Learning

- Python
- PyTorch
- CLIP
- MVPDR

## Backend

- FastAPI
- Pydantic
- SQLAlchemy

## Frontend

- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui

## Database

- PostgreSQL

## Cache

- Redis

## Vector Database

- Qdrant

## LLM

Initially:

- Ollama
- Local/open-source models

The LLM must be implemented using an abstraction so another provider can be added later.

## MLOps

- MLflow

## DevOps

- Docker
- Docker Compose
- GitHub Actions

## Monitoring

- Prometheus
- Grafana

---

# 6. Architecture Principle

The project should initially use a **modular monolith architecture**.

Do not introduce unnecessary microservices.

Target architecture:

Frontend
↓
API / FastAPI
↓
Business Services
↓
ML / RAG / Agent Modules
↓
PostgreSQL + Redis + Qdrant

---

# 7. Current Repository

GitHub repository:

https://github.com/Pankaj429w63/Terramind-AI

Current local project directory:

Terramind AI/

Important existing folder:

MVPDR/

The original MVPDR research code must be preserved.

Do not unnecessarily rewrite or delete it.

---

# 8. Current Development Phase

## Current Phase

**PHASE 0 — Project Foundation and Persistent Memory Setup**

## Completed

- Project repository created.
- Git remote connected.
- Project concept finalized.
- Project title finalized.
- Research foundation identified.
- MVPDR selected as the vision baseline.
- PlantWild selected as the primary dataset.
- Advanced roadmap designed.
- Persistent project memory system started.

## Not Yet Completed

- Full repository inspection
- Dataset download
- Dataset validation
- Google Colab baseline
- Model training
- Model evaluation
- Production inference API
- Backend
- Frontend
- Database
- RAG
- Agents
- Deployment

---

# 9. Critical Rules

## Rule 1

Never fabricate:

- Accuracy
- Precision
- Recall
- F1 score
- Training results
- Dataset statistics
- Citations
- Deployment results
- Performance results

## Rule 2

Do not claim an experiment was completed unless it was actually run.

## Rule 3

Do not commit:

- Large datasets
- Large model checkpoints
- API keys
- Passwords
- .env files

## Rule 4

Do not modify the original MVPDR baseline unnecessarily.

## Rule 5

Before every major change:

1. Read this file.
2. Read CURRENT_STATUS.md.
3. Read TODO.md.
4. Read DECISIONS.md.
5. Read KNOWN_ISSUES.md.
6. Read the latest SESSION_LOG.md entry.

## Rule 6

After every meaningful development session:

- Update CURRENT_STATUS.md.
- Update TODO.md.
- Update SESSION_LOG.md.
- Update KNOWN_ISSUES.md if necessary.
- Update this file if permanent project knowledge changed.

---

# 10. Dataset Strategy

Primary dataset:

PlantWild v1.

First:

- Verify source.
- Verify version.
- Verify license.
- Inspect folder structure.
- Inspect class labels.
- Inspect splits.
- Inspect prompts.
- Validate images.
- Check corrupt files.

Do not combine multiple datasets initially.

First reproduce a baseline using the original/compatible dataset.

Only after the baseline works should additional datasets be considered.

---

# 11. Model Development Strategy

Stage 1:

Original MVPDR baseline.

Stage 2:

Run and reproduce baseline.

Stage 3:

Evaluate baseline.

Stage 4:

Save model checkpoint.

Stage 5:

Create production inference wrapper.

Stage 6:

Expose model through FastAPI.

Do not immediately modify the research model.

---

# 12. Production Inference Output

The target prediction output is:

```json
{
  "crop": "crop_name",
  "disease": "predicted_disease",
  "confidence": 0.0,
  "top_predictions": [],
  "model_version": "MVPDR-v1",
  "inference_time": 0.0
}