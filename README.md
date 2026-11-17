# RoomCraft AI

An AI-assisted 3D room planner. You furnish a room in the browser with real, freely licensed 3D
models, and a team of AI agents can redesign it for you: they plan the layout, place the furniture,
check it for collisions, look at rendered screenshots, critique the result and, when furniture is
missing, search the web for suitable models and download them, with your approval.

The project combines an interactive **React Three Fiber** scene with a multi-agent workflow built on
**LangGraph** and **FastAPI**. LLMs are accessed through **OpenRouter**, so every agent can use a
different model.

## Features

### Interactive 3D room planner

- Search free 3D models from [3d.shep.bot](https://3d.shep.bot) and add them to a personal inventory.
- Place, move and rotate furniture in the room with a gizmo (`R` rotates by 45°, `Del` removes,
  `Esc` cancels), and search the inventory by name, description or tags.
- Every downloaded model is measured automatically (with unit detection, so centimetre models get
  the right size) and the layout is saved in the browser.

### AI room design

Describe what you want in plain language, and the **Interior Designer** agent turns it into a layout.
It first writes a functional plan (requirements, zones such as "dining" or "sofa corner", anchor
pieces, walkways) and then places the furniture zone by zone.

Every proposal goes through a **deterministic layout validator** that checks room boundaries, 3D
collisions and support (nothing floats; a lamp may stand on a table, a picture must hang on a wall).
It fixes small errors itself and returns the rest to the designer with concrete repair options.
Only validated layouts are applied to your room, and "Previous layout" undoes the change.

The designer also sees **rendered screenshots** of its layout (top-down and perspective), produced
by a headless browser from the same scene components the app uses.

> Make my living room cozy and add a reading corner by the window.

### Design critique

The **Interior Critic** evaluates a room on functionality, aesthetics, use of space and fit to your
request, with scores, concrete issues and suggestions. You can ask for a critique of your own room,
or let the critic review the designer's layout before a run finishes (the designer then decides
which points to act on).

> Critique my room. Is the space used well?

### Furniture research with human approval

When the room needs furniture that is not in your inventory, the **Furniture Research** agent works
through a list of missing pieces in parallel. It searches the asset server over **MCP**, inspects
metadata and preview images, and proposes up to four candidates per piece.

The run then **pauses for your decision**, one item at a time: download a selection, ask for a second
search with feedback ("darker wood", "smaller"), or skip. Downloads start right away in the
background while you review the next item. In automatic mode, the best candidate that passes a
deterministic check is downloaded without asking.

Direct searches first check your own furniture database (semantic search in **LanceDB**) and only
go online when nothing you already have fits.

> Find a grey sofa and two floor lamps.

### Furniture database

Every downloaded model passes one pipeline: download, measure, describe with a vision LLM (category,
style, colours, materials) and index with embeddings in LanceDB. The agents use it to find what you
already own. Details: [docs/furniture-db.md](docs/furniture-db.md).

### Live workflow view

The **Overview** tab shows every run as it happens, streamed via Server-Sent Events: the agent graph
with the active node, a timeline, timers, LLM calls with model and token usage, tool calls, validation
results, screenshots and the furniture list. Runs can be aborted at any time; nothing of an aborted
run is applied.

## How the agents work together

Simplified flow (the full graph is in [docs/graph.md](docs/graph.md)):

```
request ─► Request Router ─┬─► Interior Designer ◄──► Layout Validator
                           │        │   ▲      ▲
                           │        │   │      └──► Render Scene (screenshots)
                           │        ▼   │
                           │   Interior Critic
                           │        │
                           ├────────┴─► Furniture Research ◄──► Human Approval / Asset Download
                           └─► Interior Critic (standalone critique)
```

- **Request Router** classifies a request as design, critique, furniture search or reject. Off-topic
  requests and manipulation attempts are rejected; an injection filter runs before the LLM.
- **Interior Designer** proposes layouts step by step and decides what happens next: validate,
  request furniture, request a critique, or finish.
- **Layout Validator** (no LLM) checks and corrects the geometry.
- **Render Scene** produces screenshots whenever the layout changed substantially.
- **Interior Critic** gives feedback but never changes the layout.
- **Furniture Research**, **Human Approval** and **Asset Download** handle furniture lists.

The LLM decides *what* to do; deterministic guards decide what is *allowed*. Permissions (adding or
removing furniture, online search, automatic downloads) come from your settings, every loop has a
budget, and a run can never finish with an unvalidated layout.

The exact graph, state and routing rules are in [docs/graph.md](docs/graph.md).

## Tech stack

| Area | Technology |
|---|---|
| Frontend | React, Vite, React Three Fiber / drei |
| Backend | Python 3.12, FastAPI, LangGraph (interrupts, checkpointer) |
| LLMs | OpenRouter via the OpenAI SDK, structured output and tool calling |
| Furniture search | 3d.shep.bot REST API and MCP server |
| Furniture database | LanceDB with OpenRouter embeddings |
| Rendering | Playwright (headless Chromium) rendering the app's own scene components |
| Tests | pytest (LLM, MCP and downloads are faked) |

## Getting started

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), Node.js, and an
[OpenRouter](https://openrouter.ai) API key.

**Backend** (port 8000):

```powershell
cd src
uv sync
uv run playwright install chromium   # renderer; Edge or Chrome also work
$env:OPENROUTER_API_KEY = "..."
uv run uvicorn main:app --reload --port 8000
```

**Frontend** (port 5173, proxies `/api` to the backend):

```powershell
cd frontend
npm install
npm run build   # renderer page for the agents' screenshots; repeat after changing scene components
npm run dev
```

Then open http://localhost:5173. The API docs are at http://127.0.0.1:8000/docs.

**Tests** (the rendering tests need the frontend build):

```powershell
cd src
uv run pytest
```

## Using the app

1. **Search** (right sidebar): find models, view details and add them to the inventory.
2. **Inventory**: filter your models, click "Place", then click on the floor.
3. **Design** (left sidebar): describe what you want. Options control whether the AI may add
   furniture from the inventory, remove furniture, search online, download automatically or ask you,
   how many designer steps and furniture lists it may use, and whether the critic reviews the result.
4. **Overview**: opens automatically and shows the run live, including the approval panel when a
   decision is needed.

## Project structure

```
src/                     backend (run everything from here)
  design/                design workflow: graph, state, routing, runner, router, designer, critic,
                         validator and render nodes
  research/              furniture lists: research coordinator, human approval, asset download,
                         background jobs, MCP client
  furniture/             LanceDB store, embeddings, AI enrichment, shared asset pipeline
  rendering/             headless Playwright renderer (screenshots, model measuring)
  observability/         run store, progress events, graph view
  prompts/               the agents' system prompts as Markdown files
  tests/                 pytest suite
  main.py, *_api.py      FastAPI app and routes; llm.py is the OpenRouter client
frontend/src/
  components/            scene, sidebars, design panel, run overview
  observability/         SSE run hook and run state
  render/                headless renderer page used by Render Scene
docs/                    graph and furniture database documentation
```

Static prompts live in `src/prompts/` (one file per prompt, sent exactly as written). The messages
that carry the room, layout, validation results and screenshots are built in Python by each agent.
Edit a file to change a prompt; `tests/test_prompts.py` checks that every file is loaded.

Downloaded models are stored in `src/data/inventory/`, the furniture database in
`src/data/furniture_db/` and screenshots per run in `src/data/renders/`. The room layout lives in
the browser (localStorage; the keys still use the former name `roomforge.*`).

## Limitations and Future Work

- **Persistent runs:** Implement a persistent checkpointer (e.g., SQLite) to allow paused runs to resume after a backend restart.
- **Customizable walls and floors:** Allow users and the designer agent to modify wall and floor styles, including colors, materials, and textures.
- **Flexible room layouts:** Enable users to adjust room dimensions (length, width, and height) and add architectural elements such as doors and windows.
