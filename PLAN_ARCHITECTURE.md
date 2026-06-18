# Architecture Plan: Weather Quant Dashboard

## Current State

- **Framework**: Streamlit monolithic app
- **Deployment**: Hugging Face Docker Space
- **Core assets**: Temperature models (XGBoost 9-day + LightGBM intraday A/B/C/D/E), weather data pipeline, Polymarket market integration, backtesting engine, strategy execution pipeline
- **Known pain points**: `@st.fragment` + JS bridge state sync, limited UI customization, complex interactions fighting Streamlit's architecture

## Target Architecture

```
┌─────────────────────────────────────────────────┐
│              HF Docker Space                     │
│                                                  │
│  ┌─────────────────────┐  ┌──────────────────┐  │
│  │  FastAPI Server      │  │  Streamlit App   │  │
│  │  (uvicorn :7860)     │  │  (temporary)     │  │
│  │                      │  │                  │  │
│  │  /api/predictions ◄──┼──┤  Phase 1-2 only  │  │
│  │  /api/weather      │  │                  │  │
│  │  /api/markets      │  │                  │  │
│  │  /api/strategies   │  │                  │  │
│  │  /api/backtest     │  │                  │  │
│  └────────┬────────────┘  └──────────────────┘  │
│           │                                      │
│  ┌────────▼─────────────────────────────────┐    │
│  │  Existing Python Core (unchanged)         │    │
│  │  app/services/*  models/*  execution/*   │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Phase 3 Final Architecture

```
┌───────────────────────────────────────────────┐
│            HF Docker Space                     │
│                                                │
│  ┌─────────────────┐   ┌──────────────────┐  │
│  │  FastAPI Backend │   │  Static Frontend │  │
│  │  (uvicorn)       │   │  (Nginx serve)   │  │
│  │                  │   │                  │  │
│  │  /api/*          │   │  index.html      │  │
│  │                  │   │  /assets/*       │  │
│  └────────┬─────────┘   └──────────────────┘  │
│           │                                    │
│  ┌────────▼──────────────────────────────┐    │
│  │  Python Core (services/ models/ exec) │    │
│  └───────────────────────────────────────┘    │
│                                                │
└───────────────────────────────────────────────┘
```

## Tech Stack (Final)

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend framework | FastAPI | Native Python, async support, auto-docs, Pydantic validation |
| Frontend framework | Vite + React + TypeScript | Component model, ecosystem for drag/drop & charts |
| UI toolkit | Tailwind CSS + Radix UI | Utility-first styling, accessible headless primitives |
| State management | TanStack Query (React Query) | Server state caching, background refetch, loading/error UX |
| Charts | Recharts | Declarative, React-native, good for bucket bars + PnL curves |
| Drag & drop | @dnd-kit/core | Modern, accessible, flexible |
| Async tasks | FastAPI BackgroundTasks + polling | Lightweight, no external broker needed for single-user |
| Cache | cachetools.TTLCache | Thread-safe, time-aware replacement for st.cache_data |

## Migration Phases

| Phase | Duration | Description |
|-------|----------|-------------|
| **Phase 1** | Now | Build FastAPI layer around existing Python core. Streamlit continues to work. |
| **Phase 2** | Next | Build React frontend consuming the API. Incrementally replace Streamlit views. |
| **Phase 3** | Final | Remove Streamlit dependencies. Single FastAPI + static frontend deployment. |

## Key Decisions

1. **No Next.js** — SSG/SSR is unnecessary for a dashboard. Vite + React SPA is simpler.
2. **No Celery/RQ** — Single-user app, BackgroundTasks + polling is sufficient.
3. **Backtest is async** — Long-running task, never block the event loop.
4. **Streamlit keeps running** — Low-risk migration. Two systems coexist during Phases 1-2.
5. **No trading execution** — Deferred until after core migration.
