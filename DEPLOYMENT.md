# TerraMind AI deployment runbook

Target topology: Next.js on Vercel Hobby, FastAPI + CPU inference on Render Free, and Supabase Auth/PostgreSQL/Storage. This runbook uses no paid vector service; the existing RAG layer uses Qdrant when explicitly configured and otherwise its local fallback.

## 0. Verify the public project links

- Live frontend: <https://terramind-ai-72q1.vercel.app>
- Vercel project: <https://vercel.com/pankaj429w63/terramind-ai-72q1>
- Existing Render service: <https://dashboard.render.com/web/srv-das1f90jo6nc739r5k6g/logs?t=app&r=1h>
- Supabase project: <https://supabase.com/dashboard/project/vrdrkxrowjumhikirkyd>

The frontend URL responded with HTTP 200 and an HTML title of “TerraMind AI” during this validation. Its published bundle currently embeds `http://127.0.0.1:8001` for API calls. A working frontend response alone does not verify that the hosted API is connected.

## 1. Apply Supabase migrations

Use the Supabase CLI against the supplied project:

```powershell
supabase login
supabase link --project-ref vrdrkxrowjumhikirkyd
supabase db push
```

The migration runner applies these files in timestamp order:

1. `supabase/migrations/20260905130000_terramind_schema.sql`
2. `supabase/migrations/20260926120000_auth_reviews_and_events.sql`

The schema includes profiles, diagnoses, predictions, uploaded images, reports, agent executions, chat events, expert reviews, and the private `model-artifacts` Storage bucket. RLS is enabled on user tables. Diagnosis children are scoped through their parent diagnosis; storage object policies scope the user image/report buckets by the first path segment. The model bucket is private and is accessed only by the backend service-role client.

After applying migrations, enable email/password under **Supabase → Authentication → Providers**. Use the anon/publishable key in Vercel only. Use the service-role/secret key on Render only.

## 2. Put the unchanged model in private Storage

Render Free has no persistent disk. The code downloads the existing model at process startup only when the local checkpoint path is absent, then verifies SHA-256 before loading it.

1. Copy `.env.example` to root `.env.local` and enter the correct project URL and server-only service key.
2. Install backend dependencies locally: `python -m pip install -r requirements.txt`.
3. Upload the already-trained checkpoint without editing it:

   ```powershell
   python scripts/upload_model_checkpoint.py
   ```

   The script uploads to private bucket `model-artifacts`, object `efficientnet-b0/best_model.pth`, and prints the SHA-256 value. It does not print or change the checkpoint contents.

4. Save that SHA-256 as Render’s `TERRAMIND_MODEL_SHA256` environment variable.

The script requires a valid Supabase service credential and network access. Do not use a signed URL with a short expiry as the Render model source; the server fetches by private bucket/object on every cold start.

## 3. Configure the existing Render service

The repository includes [render.yaml](render.yaml) as a Free-plan Blueprint reference. The supplied Render service already exists, so update that service instead of creating a duplicate Blueprint service unless you intentionally want a second backend.

In the existing service settings, configure:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Branch | `main` |
| Root directory | repository root (blank) |
| Build command | `python -m pip install --upgrade pip && python -m pip install -r requirements.txt` |
| Start command | `python -m uvicorn backend.app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Instance plan | Free |

Add these environment values in the Render dashboard. Render keeps secret values out of Git:

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.11.11` |
| `SUPABASE_URL` | `https://vrdrkxrowjumhikirkyd.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase server-only service-role/secret key |
| `TERRAMIND_MODEL_BUCKET` | `model-artifacts` |
| `TERRAMIND_MODEL_OBJECT` | `efficientnet-b0/best_model.pth` |
| `TERRAMIND_MODEL_SHA256` | Checksum printed by the upload script |
| `TERRAMIND_RAG_AUTO_INGEST` | `true` |
| `TERRAMIND_LOW_CONFIDENCE_THRESHOLD` | `0.35` |
| `TERRAMIND_CORS_ORIGINS` | `https://terramind-ai-72q1.vercel.app,http://localhost:3000,http://127.0.0.1:3000` |

`QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_PATH` are optional. Leave them unset to keep the local fallback. On Render Free the filesystem and local RAG index are ephemeral, so the bundled 218 documents are re-indexed at startup. Local report/review JSON fallback on Render is also temporary; use the configured Supabase tables for durable account data.

Once environment values and the private checkpoint are in place, select **Manual Deploy → Deploy latest commit** in the existing service. Confirm `/health` reports `status: ok` and `inference_available: true`; a degraded health response means the model could not be loaded.

## 4. Connect the existing Vercel project

In **Vercel → Project → Settings → Build and Deployment**:

1. Set the Git repository to `Pankaj429w63/Terramind-AI`, production branch to `main`, framework to **Next.js**, and **Root Directory** to `frontend`.
2. Keep the default Next.js build command (`npm run build`) and output settings.
3. In **Settings → Environment Variables**, set for Production (and Preview if wanted):
   - `NEXT_PUBLIC_SUPABASE_URL=https://vrdrkxrowjumhikirkyd.supabase.co`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY=<public anon/publishable key>`
   - `TERRAMIND_BACKEND_API_URL=<the exact public https://...onrender.com origin shown on the Render service page>`
4. Remove any production `NEXT_PUBLIC_API_URL` value pointing at `127.0.0.1` or `localhost`. Leave it unset in Vercel; Next.js then sends same-origin requests through rewrites to `TERRAMIND_BACKEND_API_URL`.
5. Redeploy the production branch after saving environment variables.

Do not set `SUPABASE_SERVICE_ROLE_KEY` or any `sb_secret_*` key in Vercel. The browser uses only the public anon/publishable key; API requests are proxied by Next.js to Render.

## 5. Verify the deployed integration

Run these checks after both platform deployments finish:

1. Open the Vercel live demo and use browser Network tools to confirm `/health` and `/api/*` reach the Render origin through the same-origin rewrite; no request should target localhost.
2. Check Render `GET /health`: `status=ok`, `inference_available=true`, model `EfficientNet-B0`, `num_classes=89`.
3. Check `/api/analytics/summary`, `/api/analytics/graph/training_curves`, `/api/analytics/graph/confusion_matrix`, `/api/rag/stats`, and `/api/multimodal/status`.
4. Test one image and a small batch, then run agent diagnosis and grounded chat with indexed source evidence.
5. Sign up two test users. Confirm each user sees only their own history/reports and cannot retrieve the other user’s diagnosis by ID.
6. Submit an expert review for a low-confidence result. Confirm decision, correction, notes, and timestamp are persisted. Confirm review submission does not change the model artifact.
7. Expect multimodal analysis to be unavailable until a real completed and validated multimodal checkpoint is provided.

These are post-deployment verification steps, not claims that live tests have already passed.

## Free-tier limits and operational expectations

- Render Free spins down after 15 minutes without inbound traffic and has an ephemeral filesystem. The first request after idle can take about a minute; cold starts restore the model and rebuild the local RAG index.
- Free Render should be treated as a demo tier. PyTorch inference, model restore, and RAG indexing can exceed resource or startup limits; check Render logs and memory after deployment.
- Vercel Hobby builds Next.js from the `frontend` root. The project currently has that frontend URL live, but the backend origin must be set explicitly as described above.
- Supabase free-tier account/storage/database quotas and policies apply. The app does not silently substitute an invented auth, database, review, or model result if a hosted service is unavailable.

Official platform references: [Render FastAPI deploy](https://render.com/docs/deploy-fastapi), [Render Free limitations](https://render.com/docs/free), [Render Python version](https://render.com/docs/python-version), [Render Blueprint reference](https://render.com/docs/blueprint-spec), [Vercel monorepo root directory](https://vercel.com/docs/builds/configure-a-build#root-directory), and [Supabase `db push`](https://supabase.com/docs/reference/cli/supabase-db-push).
