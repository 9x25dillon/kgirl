# Chaos RAG Julia (single-file)

- Server: `server.jl`
- Run locally: `START_SERVER=1 bash run.sh`
- Docker build: `docker build -t chaos-rag-julia .`
- Docker run: `docker run -p 8081:8081 -e DATABASE_URL=... -e OPENAI_API_KEY=... chaos-rag-julia`
- GHCR publish workflow: `.github/workflows/publish.yml`

Endpoints:
- POST `/chaos/rag/index`
- POST `/chaos/rag/query`
- POST `/chaos/telemetry`
- POST `/chaos/hht/ingest`
- GET `/chaos/graph/:id`