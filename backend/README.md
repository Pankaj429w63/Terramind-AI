Run the FastAPI backend:

```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open the Next.js frontend and upload an image.

## RAG

Ingest real local Markdown/text documents or trusted URLs:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/rag/ingest `
  -Method Post -ContentType 'application/json' `
  -Body '{"sources":[{"source":"path/to/document.md","category":"diseases"}]}'
```

Retrieve grounded chunks and their source metadata:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/rag/retrieve `
  -Method Post -ContentType 'application/json' `
  -Body '{"query":"How is this disease treated?","limit":5}'
```

Set `QDRANT_URL` and `QDRANT_API_KEY` for hosted Qdrant. `QDRANT_PATH` enables
local persistent Qdrant; otherwise development uses an explicit in-process
store and never claims retrieval when it has no indexed sources.
