# Architecture Decisions

## ADR-001 — Project Architecture

### Decision

Use a modular monolith initially.

### Reason

The project is a final-year project and should remain manageable.

### Status

Accepted.

---

## ADR-002 — Vision Model

### Decision

Use MVPDR as the research baseline.

### Reason

It is aligned with PlantWild and the existing project research.

### Status

Accepted.

---

## ADR-003 — Primary Dataset

### Decision

Start with PlantWild v1.

### Reason

It is compatible with the research baseline.

### Status

Accepted.

---

## ADR-004 — Backend

### Decision

Use FastAPI.

### Reason

Python integration with the ML pipeline is straightforward.

### Status

Accepted.

---

## ADR-005 — Frontend

### Decision

Use Next.js and TypeScript.

### Reason

Professional full-stack web development with Vercel deployment compatibility.

### Status

Accepted.

---

## ADR-006 — Vector Database

### Decision

Use Qdrant.

### Reason

Open-source and suitable for agricultural RAG.

### Status

Accepted.