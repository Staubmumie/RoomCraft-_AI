# RoomCraft_AI

3D room planner (React Three Fiber) with an AI agent workflow (LangGraph): asset search via
[3d.shep.bot](https://3d.shep.bot), a local furniture inventory, AI layout design, critique and
furniture research with human approval.

## Running

Backend (FastAPI, port 8000). The agents need `OPENROUTER_API_KEY`
(optionally `OPENROUTER_MODEL` for another default model):

```powershell
cd src
uv sync
uv run uvicorn main:app --reload --port 8000
```

Frontend (Vite, port 5173, proxies `/api` to the backend):

```powershell
cd frontend
npm install
npm run dev
```

**Screenshots for the agents**: `render_scene` opens the built renderer page (`frontend/render.html`,
which reuses the scene components) in a headless browser. Build the frontend once, and again after
changing scene components; the backend serves the build under `/renderer/`:

```powershell
cd frontend
npm run build
```

If the backend does not run on port 8000, set `RENDER_PAGE_URL` (e.g. `http://127.0.0.1:8001/renderer/render.html`).
The renderer uses Playwright's Chromium, otherwise Edge or Chrome (`uv run playwright install chromium`).
Screenshots are grouped per run in `src/data/renders/runs/<run_id>/<seq>_v<version>_<trigger>/` (with a
`manifest.json`); identical rooms are rendered once (`src/data/renders/cache/<hash>/`) and copied. The selected model must
accept images (`DESIGNER_TEXT_ONLY_FALLBACK=1` explicitly allows text-only models).

Tests (backend; the rendering tests need the frontend build):

```powershell
cd src
uv run pytest
```

Then open http://localhost:5173. API docs: http://127.0.0.1:8000/docs

## Project structure

```
src/                         backend (FastAPI, LangGraph), run everything from here
  main.py                    FastAPI app: includes the routers, serves the renderer build
  asset_api.py               client + routes for the 3d.shep.bot REST API (/api/assets)
  inventory.py               inventory service: download, verify, register (data/inventory.json)
  inventory_api.py           inventory routes (/api/inventory)
  design_api.py              workflow routes (/api/design): start, SSE events, approval, cancel
  llm.py, url_safety.py      OpenRouter client (retries, structured output), public-URL guard
  design/                    LangGraph graph: state, routing, one module per node, validator geometry
  research/                  Furniture Research agent, MCP client, preview images
  furniture/                 LanceDB store, embeddings, AI enrichment, shared asset pipeline, routes
  rendering/                 headless Playwright renderer (screenshots, model measuring), routes
  observability/             run store (events), progress events, graph view, serialization
  tests/                     pytest suite (LLM, MCP and downloads are faked)
frontend/                    React + Vite + React Three Fiber
  src/components/            scene, sidebars, design panel, overview (graph, timeline, inspector, approval)
  src/observability/         SSE run hook and event -> run state derivation
  src/render/                headless renderer page used by render_scene (render.html)
docs/                        graph and furniture database documentation
```

Browser storage keys still use the former product name (`roomforge.layout`, `roomforge.designModel`,
`roomforge.designOptions`) so existing layouts and settings keep working.

## Using the app

1. **Search** (right sidebar): find models, view details, add them to the inventory.
2. **Inventory**: "Place", then click on the floor.
3. **Edit**: click a piece to move or rotate it (gizmo); `R` rotates by 45°, `Del` removes, `Esc` cancels.
4. **AI** (left sidebar, "Design" tab): describe what you want. The request goes through the agent
   workflow below. Only validated layouts are applied to the room; "Previous layout" undoes it.
   Options: "Include more furniture" (the designer may add inventory pieces), "Allow removing furniture"
   and "Max. iterations".
5. **Overview** tab: opens automatically and shows the run live (graph, timers, timeline, details,
   pending approvals). Events stream via SSE from `/api/design/runs/{id}/events`.

## Agent workflow

See [docs/graph.md](docs/graph.md) for the diagram and the state description.

- **Orchestrator** classifies the request: `DESIGN_ROOM`, `CRITIQUE_ROOM`, `SEARCH_FURNITURE` or `REJECT`
  (off-topic requests and manipulation attempts; an injection filter runs before the LLM).
- **Render Scene** (one shared node) renders the current working layout: the first look at the room,
  after a validated substantial change (or outdated screenshots) and for the critic. The designer
  cannot request renders itself.
- **Interior Designer** proposes layouts and decides the next step: validate, request furniture,
  request a critique, or finish (only with a validated layout).
- **Layout Validator** checks boundaries and collisions deterministically, corrects small errors
  (max. 15 cm, `ValidatorConfig` in `src/design/validator.py`) and returns to the designer, through
  Render Scene when the valid layout changed substantially.
- **Interior Critic** evaluates a layout (standalone or for the designer) without changing it.
- **Furniture Research** searches via the asset server's MCP endpoint (`https://3d.shep.bot/mcp`,
  `ASSET_MCP_URL`), looks at previews and proposes 3–5 models. **Human Approval** pauses the run
  (LangGraph `interrupt()`); **Asset Download** fetches only approved models and returns to the research,
  which hands the result to its caller (the designer or the user).

Loop limits are defined in `src/design/routing.py` (`LIMITS`). A run can be aborted at any time
(Abort run in the Design tab, the Overview or the approval panel); it is never resumed and nothing of it is applied.

## Furniture database

Every downloaded model (from the search sidebar or approved by the research agent) goes through one
pipeline: download → measure → AI description (vision LLM) → LanceDB (`src/data/furniture_db/`).
Furniture Research searches this database first and only uses the external asset server when nothing
local fits. Details: [docs/furniture-db.md](docs/furniture-db.md).

- `GET /api/furniture/search?q=...&category=...` semantic search with filters
- `POST /api/furniture/reindex` index inventory items downloaded before the database existed

Downloaded models live in `src/data/inventory/<id>/`, metadata in `src/data/inventory.json`.
Placed furniture is stored in the browser (localStorage).
