Run the FastAPI backend:

```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

Then open the Next.js frontend and upload an image.

## RAG

Ingest real local Markdown/text documents or trusted URLs:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/api/rag/ingest `
  -Method Post -ContentType 'application/json' `
  -Body '{"sources":[{"source":"path/to/document.md","category":"diseases"}]}'
```

Retrieve grounded chunks and their source metadata:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/api/rag/retrieve `
  -Method Post -ContentType 'application/json' `
  -Body '{"query":"How is this disease treated?","limit":5}'
```

Set `QDRANT_URL` and `QDRANT_API_KEY` for hosted Qdrant. `QDRANT_PATH` enables
local persistent Qdrant; otherwise development uses an explicit in-process
store and never claims retrieval when it has no indexed sources.

The vector store also maintains an atomic JSON local fallback at
`outputs/rag/vector_store_memory.json`; `/api/rag/stats` reports the active
backend, point/source counts, and fallback file size. RAG answers preserve the
retrieved source title, reference, category, and citation ID. When retrieval is
empty, chat says that no supporting source was found.

Chat uses an evidence-template answer by default; it does not claim an LLM call.
To enable an OpenAI-compatible local or remote endpoint, configure these on the
backend process (never in the browser):

```powershell
$env:TERRAMIND_LLM_BASE_URL = "http://localhost:11434/v1"
$env:TERRAMIND_LLM_MODEL = "your-served-model"
# Optional for providers that require it:
$env:TERRAMIND_LLM_API_KEY = "..."
```

The provider has a bounded request timeout and must cite supplied evidence as
`[S1]`-style IDs. Provider failures return the source-template answer and expose
the fallback mode and error class. `GET /api/llm/status` reports configuration
without disclosing the key. The default template works without external
services.
