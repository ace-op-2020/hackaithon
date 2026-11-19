# hackaithon

This repository contains two parts:

- `mcp-server/` – a Python (FastAPI + Vertex AI) MCP server that turns natural-language requests into action steps.
- `my-react-app/` – a React app (built with `react-scripts` and `craco`) that talks to the server.

## Local development

### Prerequisites

- Python 3.8+ (the minimum stated in `mcp-server/README.md`)
- Node.js and npm (for `my-react-app/`)
- Google Cloud access to Vertex AI. `mcp-server/vertex_ai_service.py` expects a service-account key file named `svc-hackathon-prod15-534bc641841c.json` in the repository root (it is not committed; obtain it separately).

### MCP server (`mcp-server/`)

1. Create and activate a virtual environment:

   ```bash
   cd mcp-server
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the server (listens on port 8000):

   ```bash
   python main.py
   # or, equivalently:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   Check it at `http://localhost:8000/` or `http://localhost:8000/health`.

See `mcp-server/README.md` for the API endpoints and request/response formats.

### React app (`my-react-app/`)

1. Install dependencies:

   ```bash
   cd my-react-app
   npm install
   ```

2. Create your environment file from the example and fill in your values (the variables are listed in `env.example`):

   ```bash
   cp env.example .env
   ```

3. Start the dev server:

   ```bash
   npm start
   ```

   The MCP server's CORS settings allow the origins `http://localhost:3000` and `http://localhost:3001`.

### Tests

- **React app:** `npm test` (runs `craco test`) from `my-react-app/`.
- **MCP server:** the `test_*.py` files in `mcp-server/` (`test_improved_vertex.py`, `test_rag_implementation.py`, `test_responsible_ai.py`) are standalone scripts, not a pytest suite. They call Vertex AI, so they need valid Google Cloud credentials. Run them from `mcp-server/` with the virtualenv active, for example:

  ```bash
  python test_responsible_ai.py
  ```

  Note: the `test` script in `mcp-server/package.json` points at a `tests/` directory that does not exist in the repo.
