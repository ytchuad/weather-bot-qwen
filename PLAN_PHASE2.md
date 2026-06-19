# Phase 2: React Frontend

## Status

Completed initial implementation. The frontend is a Vite + React + TypeScript SPA that consumes the FastAPI API from Phase 1.

## Tech Stack

| Layer | Package | Purpose |
|-------|---------|---------|
| Framework | Vite + React 19 + TypeScript | SPA build and type safety |
| API state | TanStack Query | Server-state cache, background refresh, loading/error handling |
| Routing | React Router v7 | `/` Hub, `/strategies` |
| Styling | Tailwind CSS v4 | Utility-first dark glass UI |
| Charts | Recharts | Bucket probability comparison |
| Drag/drop | @dnd-kit/core + sortable | Model card drag reorder |
| Icons | lucide-react | Navigation icons |

## Project Structure

```
app/frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── index.css
    ├── api/
    │   └── client.ts        # API wrapper functions
    ├── components/
    │   ├── Layout.tsx       # Top nav
    │   ├── WeatherNow.tsx   # Current weather card
    │   ├── ModelGrid.tsx    # Model cards + drag/reorder
    │   └── BucketChart.tsx  # Recharts bucket comparison
    ├── pages/
    │   ├── Hub.tsx          # Main dashboard
    │   └── Strategies.tsx   # Strategy suggestion list
    └── types/
        └── index.ts
```

## Local Development

Run FastAPI backend:

```bash
uvicorn app.api.server:app --host 127.0.0.1 --port 7860
```

Run frontend dev server:

```bash
cd app/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open: <http://localhost:5173>

The Vite dev server proxies `/api/*` to `http://127.0.0.1:7860`.

## Build

```bash
cd app/frontend
npm run build
```

Output: `app/frontend/dist/`

## Production Serving

FastAPI automatically mounts `app/frontend/dist` if it exists:

```python
_frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(...))
```

When built, `APP_MODE=api` serves:

- `/` → React frontend
- `/docs` → Swagger UI
- `/api/*` → API endpoints

## Current Pages

### `/` Hub

- Date selector (`input[type="date"]`)
- Current weather card (`/api/weather/now?date=...`)
- Event lookup (`/api/markets/events?date=...`)
- Event market prices (`/api/markets/event/{slug}`)
- Model prediction cards (`/api/predictions?date=TODAY&is_min_temp=false`)
- Drag/reorder model cards with @dnd-kit
- Bucket chart with Model vs Market probabilities

### `/strategies`

- Calls `/api/strategies/suggest`
- Groups suggestions into "Signals" and "Pass"
- Displays model probability, market price, edge, Kelly fraction

## Known Gaps / Next Iteration

1. **Strategy inputs** — Capital and Kelly fraction are hardcoded in the UI; should be editable.
2. **Backtest UI** — Not yet implemented; API endpoints exist.
3. **Selected model persistence** — Drag reorder currently stores only local order; should persist/order sync via API if desired.
4. **TMAX/TMIN selector** — Hub currently uses TMAX (`is_min_temp=false`); should let user switch market type.
