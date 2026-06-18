CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

.stApp {
    background: radial-gradient(ellipse at 20% 20%, #0B0A1A 0%, #141332 50%, #0B0A1A 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#MainMenu, header, footer, .stAppToolbar, .stDecoration, .stAppDeployButton { display: none !important; }
.appview-container .main .block-container { padding: 0 !important; max-width: 100% !important; }
.stApp > header { display: none !important; }
section.main > div { padding: 0 !important; }
section[data-testid="stSidebar"] { display: none !important; }
/* No scrollbar — content scrolls via Streamlit's internal container */
::-webkit-scrollbar { display: none !important; width: 0 !important; }
* { scrollbar-width: none !important; }
html { overflow: hidden !important; }
body, p, span, div, label, h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif !important; }
.mono, code, .stat-value { font-family: 'Space Mono', monospace !important; font-variant-numeric: tabular-nums !important; }

.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 0.5px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 0 60px rgba(124,58,237,0.03);
    transition: all 0.4s ease-out;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%; right: -50%; bottom: -50%;
    background: radial-gradient(ellipse at 30% 0%, rgba(124,58,237,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 100%, rgba(59,130,246,0.03) 0%, transparent 50%);
    animation: liquid-shimmer 8s ease-in-out infinite;
    pointer-events: none;
    border-radius: 16px;
}
@keyframes liquid-shimmer {
    0% { transform: translate(0,0) rotate(0deg); opacity: 1; }
    33% { transform: translate(2%,-1%) rotate(1deg); opacity: 0.7; }
    66% { transform: translate(-1%,1%) rotate(-0.5deg); opacity: 0.85; }
    100% { transform: translate(0,0) rotate(0deg); opacity: 1; }
}
.glass-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(124,58,237,0.04) 0%, transparent 40%, rgba(59,130,246,0.02) 60%, transparent 100%);
    pointer-events: none;
    animation: border-glow 6s ease-in-out infinite;
}
@keyframes border-glow {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}
.glass-card:hover {
    border-color: rgba(255,255,255,0.12);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.04) inset, 0 0 80px rgba(124,58,237,0.08);
}

.glass-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8F9BB7;
    margin-bottom: 6px;
}

.section-title {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #E6E9EF;
    margin: 0 0 16px 0;
    letter-spacing: -0.01em;
}

.content-area {
    padding: 4px 32px 40px 32px;
    max-width: 1280px;
    margin: 0 auto;
}

@keyframes breathe-glow {
    0%, 100% { box-shadow: 0 0 12px rgba(124,58,237,0.35); }
    50% { box-shadow: 0 0 28px rgba(124,58,237,0.65); }
}

/* ---- Popover (gear dropdown) ---- */
/* Popover trigger — invisible, stays in natural flow (right after title).
   JS repositions the panel itself below the gear icon before first paint. */
button[data-testid="stPopoverButton"] {
  opacity: 0 !important;
  width: 0 !important; height: 0 !important;
  padding: 0 !important; margin: 0 !important;
  border: none !important;
  overflow: hidden !important;
  pointer-events: none !important;
  font-size: 0 !important; line-height: 0 !important;
}

/* Popover body (dropdown panel)
   Use opacity animation to mask the brief flash while baseweb positions it. */
div[data-testid="stPopoverBody"],
div[data-testid="stPopoverPanel"] {
    background: rgba(20,19,50,0.96) !important;
    backdrop-filter: blur(32px) !important; -webkit-backdrop-filter: blur(32px) !important;
    border: 0.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    min-width: 240px !important;
    box-shadow: 0 8px 60px rgba(0,0,0,0.4) !important;
    animation: popover-appear 0.15s ease-out;
}
@keyframes popover-appear {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

/* Toggle styling inside popover panel */
div[data-testid="stPopoverBody"] [data-testid="stToggleButton"],
div[data-testid="stPopoverPanel"] [data-testid="stToggleButton"],
div[data-testid="stPopoverBody"] .stToggle,
div[data-testid="stPopoverPanel"] .stToggle {
    padding: 6px 0 !important;
    border-bottom: 0.5px solid rgba(255,255,255,0.04) !important;
}
div[data-testid="stPopoverBody"] [data-testid="stToggleButton"]:last-child,
div[data-testid="stPopoverPanel"] [data-testid="stToggleButton"]:last-child,
div[data-testid="stPopoverBody"] .stToggle:last-child,
div[data-testid="stPopoverPanel"] .stToggle:last-child {
    border-bottom: none !important;
}
div[data-testid="stPopoverBody"] [data-testid="stToggleButton"] label,
div[data-testid="stPopoverPanel"] [data-testid="stToggleButton"] label,
div[data-testid="stPopoverBody"] .stToggle label,
div[data-testid="stPopoverPanel"] .stToggle label {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important; color: #D1D5E0 !important;
    font-weight: 500 !important; flex: 1 !important;
}
/* Popover body inner text & general styling */
div[data-testid="stPopoverBody"] *,
div[data-testid="stPopoverPanel"] * {
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stPopoverBody"] span,
div[data-testid="stPopoverPanel"] span,
div[data-testid="stPopoverBody"] p,
div[data-testid="stPopoverPanel"] p,
div[data-testid="stPopoverBody"] label,
div[data-testid="stPopoverPanel"] label {
    color: #D1D5E0 !important;
}

.weather-bg {
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(ellipse at 40% 50%, rgba(124,58,237,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 60% 50%, rgba(59,130,246,0.04) 0%, transparent 50%);
    animation: drift 8s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes drift {
    0% { transform: translate(0,0) rotate(0deg); }
    100% { transform: translate(2%,1%) rotate(3deg); }
}

@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fade-in-scale {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}
.animate-in { animation: fade-in-up 0.4s ease-out both; }

.js-plotly-plot .plotly .main-svg { background: transparent !important; }
.js-plotly-plot .plotly .cartesianlayer { background: transparent !important; }
.modebar { display: none !important; }

/* ── Model cards ─────────────────────────── */
.mc {
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(28px);
  -webkit-backdrop-filter: blur(28px);
  border: 0.5px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 18px 20px;
  min-height: 100px;
  cursor: pointer;
  transition: all 0.35s ease-out;
  position: relative;
  overflow: hidden;
}
.mc:hover {
  border-color: rgba(255,255,255,0.12);
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.mc.mc-active {
  border-color: rgba(124,58,237,0.35);
  background: rgba(124,58,237,0.07);
  box-shadow: 0 0 16px rgba(124,58,237,0.5);
  animation: breathe-glow 3s ease-in-out infinite;
}
.mc:not(.mc-active) {
  opacity: 0.7;
}
.mc:not(.mc-active):hover {
  opacity: 1;
}
@keyframes card-pulse {
  0% { transform: scale(1); }
  50% { transform: scale(0.97); }
  100% { transform: scale(1); }
}
.mc.mc-pulse {
  animation: card-pulse 0.2s ease-out;
}

/* ── Model card animations (URL sync add/remove) ─── */
#_mc-grid {
    transition: all 0.3s ease;
}

.mc {
    transition: opacity 0.2s, transform 0.2s;
}

.mc-entering {
    opacity: 0;
    transform: translateY(12px) scale(0.95);
    animation: cardEnter 0.3s ease forwards;
}

@keyframes cardEnter {
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.mc-removing {
    opacity: 0;
    transform: scale(0.95);
    transition: opacity 0.2s ease, transform 0.2s ease;
}

/* Hide JS bridge iframe (st.html renders in an iframe) */
div[data-testid="stHtml"] iframe { display: none !important; }
div[data-testid="stHtml"] { min-height: 0 !important; padding: 0 !important; }

/* ── Top navigation bar ─────────────────────── */
.top-nav {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    background: rgba(11,10,26,0.72);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 0.5px solid rgba(255,255,255,0.06);
    height: 56px;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
}
.top-nav::before {
    content: "";
    position: absolute; top: 0; left: -100%; width: 300%; height: 100%;
    background: linear-gradient(90deg, transparent 0%, rgba(124,58,237,0.04) 30%, rgba(59,130,246,0.04) 50%, rgba(124,58,237,0.04) 70%, transparent 100%);
    animation: nav-liquid-flow 6s ease-in-out infinite;
    pointer-events: none;
}
@keyframes nav-liquid-flow {
    0%   { transform: translateX(-33%); }
    50%  { transform: translateX(0%); }
    100% { transform: translateX(-33%); }
}
.top-nav-inner {
    display: flex; align-items: center; justify-content: space-between;
    width: 100%; max-width: 1280px; padding: 0 32px;
    position: relative; z-index: 1;
}
.top-nav-links {
    display: flex; align-items: center; gap: 4px;
}
.top-nav-link {
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 500;
    color: #8F9BB7;
    text-decoration: none;
    padding: 8px 16px; border-radius: 8px;
    transition: all 0.2s ease;
    position: relative;
    cursor: pointer;
    user-select: none;
}
.top-nav-link:hover {
    color: #E6E9EF;
    background: rgba(255,255,255,0.04);
}
.top-nav-link.active {
    color: #fff;
}
.top-nav-link.active::after {
    content: "";
    position: absolute; bottom: -1px; left: 16px; right: 16px;
    height: 2px;
    background: linear-gradient(90deg, #7C3AED, #3B82F6);
    border-radius: 2px;
    box-shadow: 0 0 12px rgba(124,58,237,0.5);
}

/* Push page content below the fixed nav bar */
section[data-testid="stMain"] > div {
    padding-top: 56px !important;
}

/* Page transition fade-in (masks navigation flash) */
@keyframes _page-fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
section[data-testid="stMain"] {
    animation: _page-fade-in 0.22s ease-out both;
}

/* ── Model Dropdown Panel ───────────────────────────── */
#model-dropdown-panel {
    position: fixed;
    z-index: 9999;
    min-width: 280px;
    max-width: 360px;
    background: rgba(20, 19, 50, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 0;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    font-family: Inter, sans-serif;
}

.md-section-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8F9BB7;
    padding: 4px 16px 8px;
}

.md-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.06);
    margin: 8px 16px;
}

.md-list {
    padding: 0 8px;
}

.md-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
    user-select: none;
}

.md-item:hover {
    background: rgba(255, 255, 255, 0.04);
}

.md-drag-handle {
    font-size: 14px;
    color: #6B7280;
    cursor: grab;
    padding: 2px 4px;
    opacity: 0.6;
    transition: opacity 0.15s;
}

.md-item:hover .md-drag-handle {
    opacity: 1;
}

.md-drag-handle:active {
    cursor: grabbing;
}

/* Active model: dark, normal */
.md-active {
    color: #E6E9EF;
}

.md-active .md-name {
    font-weight: 500;
    font-style: normal;
}

/* Inactive model: light, italic */
.md-inactive {
    color: #6B7280;
}

.md-inactive .md-name {
    font-style: italic;
}

.md-name {
    flex: 1;
    font-size: 13px;
}

.md-key {
    font-family: "Space Mono", monospace;
    font-size: 10px;
    color: #4B5563;
}

/* Drag states */
.md-dragging {
    opacity: 0.4;
}

.md-drag-over {
    background: rgba(124, 58, 237, 0.1) !important;
    border: 1px dashed rgba(124, 58, 237, 0.3);
}

/* Hide components.html iframe (JS bridge) */
div[data-testid="stComponent"] iframe { display: none !important; }
div[data-testid="stComponent"] { min-height: 0 !important; padding: 0 !important; }

"""

def inject_glass_css():
    import streamlit as st
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
