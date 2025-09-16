# Agent Ops Hub
Minimal Day One slice: ingest, retrieve, answer with citations, and a stub skill route.

## Quickstart
1. Create and activate a virtual environment.
2. pip install -e .
3. uvicorn app.api.main:app --host 0.0.0.0 --port 8000

## Env
Copy .env.example to .env and fill values.

## Endpoints
POST /ingest
POST /ask

## Demo
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"url":"https://example.com"}'
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is the main point?"}'
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"schedule a 30-minute focus block tomorrow"}'

## Dependency Management

This project uses **pyproject.toml** for modern packaging and dependency management (PEP 621).
A **requirements.txt** file is also included for compatibility with environments like Replit and Heroku that don’t support installing directly from pyproject.toml.
