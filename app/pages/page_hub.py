from __future__ import annotations

import json as _json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ..components.top_nav import mark_refreshed
from ..config import MODEL_KEYS, MODEL_LABELS, TMAX_BUCKETS, TMIN_BUCKETS
from ..services.market_service import fetch_event_markets
from ..services.model_service import run_all_models
from ..services.weather_service import (
    compute_rain_kwargs,
    fetch_hko_data,
    get_intraday_state,
    hkt_now,
)
from ..services.today_event_resolver import resolve_today_event
from ..state import AppState

_MODELS_KEY = "hub_selected_models"
_SELECTED_KEY = "hub_selected_model"
_DEFAULT_MODELS = ["baseline", "rain_nowcast", "model_a", "model_2a", "model_2a_v2"]


def _init_models(all_results: dict):
    if _MODELS_KEY not in st.session_state:
        avail = [k for k in all_results if k not in ("9d", "aws", "_intraday_error")]
        st.session_state[_MODELS_KEY] = [m for m in _DEFAULT_MODELS if m in avail][:4]
    if _SELECTED_KEY not in st.session_state:
        sm = st.session_state.get(_MODELS_KEY, [])
        st.session_state[_SELECTED_KEY] = sm[0] if sm else None


def _model_options(all_results: dict) -> list[str]:
    return [k for k in all_results if k not in ("9d", "aws", "_intraday_error")]


def _active_model(selected_models: list[str], all_results: dict) -> str:
    m = st.session_state.get(_SELECTED_KEY)
    if m and m in all_results:
        return m
    if m and m in selected_models:
        return m
    if selected_models:
        return selected_models[0]
    return next((k for k in all_results if k != "_intraday_error"), "")



def _temp_card_html(temp: float, humidity: float, max_obs: float | None, min_obs: float | None) -> str:
    mx = f"{max_obs:.1f}&deg;C" if max_obs is not None else "--"
    mn = f"{min_obs:.1f}&deg;C" if min_obs is not None else "--"
    return f"""
    <div class="glass-card" style="position:relative;overflow:hidden;min-height:150px;">
      <div class="weather-bg"></div>
      <div style="position:relative;z-index:1;">
        <div class="glass-label">Current Temperature</div>
        <div style="display:flex;align-items:baseline;gap:6px;margin-top:4px;">
          <span class="mono" style="font-size:48px;font-weight:700;color:#fff;line-height:1;">{temp:.1f}</span>
          <span style="font-size:22px;color:#8F9BB7;font-family:Inter;">&deg;C</span>
        </div>
        <div style="display:flex;gap:20px;margin-top:8px;">
          <div><span class="glass-label">Max</span><br><span class="mono" style="color:#F59E0B;font-size:14px;">{mx}</span></div>
          <div><span class="glass-label">Min</span><br><span class="mono" style="color:#22D3EE;font-size:14px;">{mn}</span></div>
          <div><span class="glass-label">Humidity</span><br><span class="mono" style="color:#D1D5E0;font-size:14px;">{humidity:.0f}%</span></div>
        </div>
      </div>
    </div>
    """


def _rain_card_html(cumul_mm: float, rain_60m: float, rain_120m: float, forecast_mm: float) -> str:
    bar = min(cumul_mm / 50 * 100, 100)
    return f"""
    <div class="glass-card" style="position:relative;min-height:150px;">
      <div style="position:relative;z-index:1;">
        <div class="glass-label">Cumulative Rainfall</div>
        <div style="display:flex;align-items:baseline;gap:6px;margin-top:4px;">
          <span class="mono" style="font-size:40px;font-weight:700;color:#3B82F6;line-height:1;">{cumul_mm:.1f}</span>
          <span style="font-size:16px;color:#8F9BB7;font-family:Inter;">mm today</span>
        </div>
        <div style="display:flex;gap:24px;margin-top:10px;">
          <div><span class="glass-label">60m</span><br><span class="mono" style="color:#D1D5E0;font-size:13px;">{rain_60m:.1f}mm</span></div>
          <div><span class="glass-label">120m</span><br><span class="mono" style="color:#D1D5E0;font-size:13px;">{rain_120m:.1f}mm</span></div>
          <div><span class="glass-label">Forecast</span><br><span class="mono" style="color:#A78BFA;font-size:13px;">{forecast_mm:.1f}mm</span></div>
        </div>
        <div style="margin-top:10px;height:4px;border-radius:4px;background:rgba(255,255,255,0.06);overflow:hidden;">
          <div style="height:100%;width:{bar:.0f}%;border-radius:4px;background:linear-gradient(90deg,#7C3AED,#3B82F6);transition:width 0.6s;"></div>
        </div>
      </div>
    </div>
    """


def _bucket_rows_html(markets: list, probs: dict) -> str:
    rows = ""
    for m in markets:
        b = m["bucket"]
        mp = float(m.get("yes_price", 0.5))
        mpp = float(probs.get(b, 0))
        diff = mpp - mp
        ds = f"+{diff*100:.1f}%" if diff >= 0 else f"{diff*100:.1f}%"
        c = "#22C55E" if diff >= 0 else "#EF4444"
        poly_pct = f"{mp*100:.0f}"
        model_pct = f"{mpp*100:.0f}"
        rows += (
            '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;'
            'border-bottom:0.5px solid rgba(255,255,255,0.04);">'
            '<div class="mono" style="width:40px;font-size:11px;color:#8F9BB7;text-align:right;">' + b + '</div>'
            '<div style="flex:1;display:flex;flex-direction:column;gap:4px;">'
            '<div style="display:flex;align-items:center;gap:4px;">'
            '<span style="font-size:7px;color:#6B7280;font-family:Inter;letter-spacing:0.08em;width:30px;text-align:right;">POLY</span>'
            '<div style="flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,0.04);overflow:hidden;">'
            '<div style="height:100%;width:' + poly_pct + '%;border-radius:3px;background:#A78BFA;transition:width 0.4s;"></div></div>'
            '<span class="mono" style="width:30px;font-size:9px;color:#A78BFA;text-align:right;">' + poly_pct + '%</span>'
            '</div>'
            '<div style="display:flex;align-items:center;gap:4px;">'
            '<span style="font-size:7px;color:#6B7280;font-family:Inter;letter-spacing:0.08em;width:30px;text-align:right;">MOD</span>'
            '<div style="flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,0.04);overflow:hidden;">'
            '<div style="height:100%;width:' + model_pct + '%;border-radius:3px;background:' + c + ';transition:width 0.4s;"></div></div>'
            '<span class="mono" style="width:30px;font-size:9px;color:' + c + ';text-align:right;">' + model_pct + '%</span>'
            '</div>'
            '</div>'
            '<div class="mono" style="width:38px;font-size:10px;color:' + c + ';text-align:center;">' + ds + '</div>'
            '</div>'
        )
    return rows


def _chart(df_today: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    t, x = df_today["temp"], df_today["datetime"]
    fig.add_trace(go.Scatter(x=x, y=t, mode="lines", line=dict(color="#7C3AED", width=1.5),
                             fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
                             hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.1f}&deg;C<extra></extra>", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=t.rolling(3, min_periods=1).mean(), mode="lines",
                             line=dict(color="#3B82F6", width=1), fill="tonexty",
                             fillcolor="rgba(59,130,246,0.06)",
                             hovertemplate="Smooth: %{y:.1f}&deg;C<extra></extra>", showlegend=False))
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False,
                     range=[x.iloc[0], x.iloc[-1] + pd.Timedelta(minutes=30)])
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0), height=200, hovermode="x unified",
                      hoverlabel=dict(bgcolor="rgba(20,19,50,0.95)", bordercolor="rgba(255,255,255,0.1)",
                                      font=dict(family="Space Mono,monospace", size=11, color="#E6E9EF")),
                      xaxis=dict(rangeselector=dict(
                          buttons=[dict(count=1, label="1H", step="hour", stepmode="backward"),
                                   dict(count=6, label="6H", step="hour", stepmode="backward"),
                                   dict(count=1, label="1D", step="day", stepmode="backward")],
                          bgcolor="rgba(255,255,255,0.04)", activecolor="rgba(124,58,237,0.3)",
                          bordercolor="rgba(255,255,255,0.08)", font=dict(color="#8F9BB7", size=10), x=0, y=1),
                          type="date"))
    return fig





_JS_BRIDGE = """<script>
(function S(){
  var doc = window.parent.document;
  if (doc.body.hasAttribute('data-mc-wired')) return;
  doc.body.setAttribute('data-mc-wired', '1');

  var dropdownOpen = false;
  var dragSrcEl = null;

  /* ── Dropdown helpers ─────────────────────────────── */
  function toggleDropdown() {
    var panel = doc.querySelector('#model-dropdown-panel');
    if (!panel) return;
    dropdownOpen = !dropdownOpen;
    panel.style.display = dropdownOpen ? 'block' : 'none';
    if (dropdownOpen) positionDropdown();
  }

  function closeDropdown() {
    dropdownOpen = false;
    var panel = doc.querySelector('#model-dropdown-panel');
    if (panel) panel.style.display = 'none';
  }

  function positionDropdown() {
    var gb = doc.querySelector('._gear-btn');
    var panel = doc.querySelector('#model-dropdown-panel');
    if (!gb || !panel) return;
    var gr = gb.getBoundingClientRect();
    panel.style.position = 'fixed';
    panel.style.top = (gr.bottom + 8) + 'px';
    panel.style.right = (window.innerWidth - gr.right) + 'px';
    panel.style.left = 'auto';
    panel.style.bottom = 'auto';
  }

  /* ── Build dropdown from JSON ─────────────────────── */
  function buildDropdown() {
    var dataEl = doc.querySelector('#model-dropdown-data');
    var panel = doc.querySelector('#model-dropdown-panel');
    if (!dataEl || !panel) return;

    var data;
    try { data = JSON.parse(dataEl.textContent); } catch(e) { return; }

    var activeList = panel.querySelector('#md-active-list');
    var inactiveList = panel.querySelector('#md-inactive-list');
    if (!activeList || !inactiveList) return;

    activeList.innerHTML = '';
    inactiveList.innerHTML = '';

    data.active.forEach(function(m) {
      var el = doc.createElement('div');
      el.className = 'md-item md-active';
      el.setAttribute('data-key', m.key);
      el.setAttribute('draggable', 'true');
      el.innerHTML = '<span class="md-drag-handle">⋮⋮</span>'
        + '<span class="md-name">' + m.label + '</span>'
        + '<span class="md-key">[' + m.key + ']</span>';
      activeList.appendChild(el);
    });

    data.inactive.forEach(function(m) {
      var el = doc.createElement('div');
      el.className = 'md-item md-inactive';
      el.setAttribute('data-key', m.key);
      el.innerHTML = '<span class="md-drag-handle" style="visibility:hidden;">⋮⋮</span>'
        + '<span class="md-name">' + m.label + '</span>'
        + '<span class="md-key">[' + m.key + ']</span>';
      inactiveList.appendChild(el);
    });
  }

  /* ── Grid helpers ───────────────────────────────── */
  function adjustGridColumns() {
    var grid = doc.querySelector('#_mc-grid');
    if (!grid) return;
    var count = grid.querySelectorAll('.mc').length;
    count = Math.max(count, 1);
    grid.style.gridTemplateColumns = 'repeat(' + count + ', 1fr)';
  }

  function addCard(key) {
    var grid = doc.querySelector('#_mc-grid');
    var cardDataEl = doc.querySelector('#card-data');
    if (!grid || !cardDataEl) return;

    var cardData;
    try { cardData = JSON.parse(cardDataEl.textContent); } catch(e) { return; }

    var cardHtml = cardData[key];
    if (!cardHtml) return;

    var wrapper = doc.createElement('div');
    wrapper.innerHTML = cardHtml;
    var newCard = wrapper.firstChild;
    newCard.classList.add('mc-entering');
    grid.appendChild(newCard);

    requestAnimationFrame(function() {
      newCard.classList.remove('mc-entering');
    });

    adjustGridColumns();
  }

  function removeCard(key) {
    var card = doc.querySelector('.mc[data-key="' + key + '"]');
    if (!card) return;

    card.classList.add('mc-removing');
    setTimeout(function() {
      if (card.parentNode) card.parentNode.removeChild(card);
      adjustGridColumns();
    }, 200);
  }

  /* ── Drag & Drop (event delegation on active list) ── */
  function initDragDelegation() {
    var container = doc.querySelector('#md-active-list');
    if (!container) return;

    container.addEventListener('dragstart', function(e) {
      var item = e.target.closest('.md-item');
      if (!item) return;
      dragSrcEl = item;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', item.getAttribute('data-key'));
      item.classList.add('md-dragging');
    });

    container.addEventListener('dragend', function(e) {
      var item = e.target.closest('.md-item');
      if (item) item.classList.remove('md-dragging');
      container.querySelectorAll('.md-item').forEach(function(it) {
        it.classList.remove('md-drag-over');
      });
      dragSrcEl = null;
      sendReorder();
    });

    container.addEventListener('dragover', function(e) {
      if (e.preventDefault) e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      var overItem = e.target.closest('.md-item');
      if (overItem && overItem !== dragSrcEl) {
        container.querySelectorAll('.md-item').forEach(function(it) {
          it.classList.remove('md-drag-over');
        });
        overItem.classList.add('md-drag-over');
      }
      return false;
    });

    container.addEventListener('dragleave', function(e) {
      var item = e.target.closest('.md-item');
      if (item) item.classList.remove('md-drag-over');
    });

    container.addEventListener('drop', function(e) {
      if (e.stopPropagation) e.stopPropagation();
      var overItem = e.target.closest('.md-item');
      if (overItem && dragSrcEl && overItem !== dragSrcEl) {
        var srcIdx = Array.from(container.children).indexOf(dragSrcEl);
        var dstIdx = Array.from(container.children).indexOf(overItem);
        if (srcIdx < dstIdx) {
          container.insertBefore(dragSrcEl, overItem.nextSibling);
        } else {
          container.insertBefore(dragSrcEl, overItem);
        }
      }
      return false;
    });
  }

  function buildCardGrid() {
    var grid = doc.querySelector('#_mc-grid');
    var cardDataEl = doc.querySelector('#card-data');
    if (!grid || !cardDataEl) return;
    var cardData;
    try { cardData = JSON.parse(cardDataEl.textContent); } catch(e) { return; }
    var activeList = doc.querySelector('#md-active-list');
    if (!activeList) return;
    var selKey = null;
    var curActive = grid.querySelector('.mc-active');
    if (curActive) selKey = curActive.getAttribute('data-key');
    var keys = [];
    activeList.querySelectorAll('.md-item').forEach(function(el) {
      var k = el.getAttribute('data-key');
      if (k && cardData[k]) keys.push(k);
    });
    grid.innerHTML = '';
    keys.forEach(function(key) {
      var wrapper = doc.createElement('div');
      wrapper.innerHTML = cardData[key];
      var card = wrapper.firstChild;
      if (key === selKey) card.classList.add('mc-active');
      grid.appendChild(card);
    });
    adjustGridColumns();
  }

  function sendReorder() {
    buildCardGrid();
  }

  function toggleModel(key) {
    var item = doc.querySelector('.md-item[data-key="' + key + '"]');
    if (!item) return;
    var isActive = item.classList.contains('md-active');
    var activeList = doc.querySelector('#md-active-list');
    var inactiveList = doc.querySelector('#md-inactive-list');
    if (isActive) {
      removeCard(key);
      item.classList.remove('md-active');
      item.classList.add('md-inactive');
      item.setAttribute('draggable', 'false');
      var handle = item.querySelector('.md-drag-handle');
      if (handle) handle.style.visibility = 'hidden';
      if (inactiveList) inactiveList.appendChild(item);
    } else {
      addCard(key);
      item.classList.remove('md-inactive');
      item.classList.add('md-active');
      item.setAttribute('draggable', 'true');
      var handle = item.querySelector('.md-drag-handle');
      if (handle) handle.style.visibility = 'visible';
      if (activeList) activeList.appendChild(item);
    }
  }

  function syncToServer(selectedKey) {
    var activeList = doc.querySelector('#md-active-list');
    if (!activeList) return;
    var keys = [];
    activeList.querySelectorAll('.md-item').forEach(function(el) {
      var k = el.getAttribute('data-key');
      if (k) keys.push(k);
    });
    var url = new URL(window.parent.location);
    url.searchParams.set('ml_order', keys.join(','));
    url.searchParams.set('ml_sel', selectedKey);
    window.parent.history.replaceState({}, '', url);
    var btns = doc.querySelectorAll('div[data-testid="stButton"] button');
    var btn = null;
    btns.forEach(function(b) { if (b.textContent.trim() === 'Sync') btn = b; });
    if (btn) btn.click();
  }

  /* ── MutationObserver: rebuild on fragment rerun ──── */
  var dataObs = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      mutation.addedNodes.forEach(function(node) {
        if (node.nodeType === 1 && (
          node.id === 'model-dropdown-data' ||
          (node.querySelector && node.querySelector('#model-dropdown-data'))
        )) {
          buildDropdown();
          adjustGridColumns();
        }
      });
    });
  });
  dataObs.observe(doc.body, { childList: true, subtree: true });

  /* ── Event delegation ───────────────────────────── */
  doc.body.addEventListener('click', function(e){
    /* 1. Dropdown item click (toggle on/off) */
    var mdItem = e.target.closest('.md-item');
    if (mdItem && e.target.closest('#model-dropdown-panel')) {
      if (e.target.closest('.md-drag-handle')) return;
      var k = mdItem.getAttribute('data-key');
      if (k) toggleModel(k);
      return;
    }

    /* 2. Gear icon → toggle dropdown */
    var gb = e.target.closest('._gear-btn');
    if (gb) {
      e.preventDefault();
      e.stopPropagation();
      toggleDropdown();
      return;
    }

    /* 3. Click outside dropdown → close */
    if (dropdownOpen && !e.target.closest('#model-dropdown-panel')) {
      closeDropdown();
    }

    /* 4. Model card click — update selected + sync to server */
    var card = e.target.closest('.mc');
    if (!card) return;
    var k = card.getAttribute('data-key');
    if (!k) return;

    var freshBd = doc.querySelector('#bucket-data');
    var data = freshBd ? JSON.parse(freshBd.textContent) : {};

    doc.querySelectorAll('.mc').forEach(function(c){
      c.classList.remove('mc-active');
      c.classList.remove('mc-pulse');
    });
    card.classList.add('mc-active');
    card.classList.add('mc-pulse');
    setTimeout(function(){ card.classList.remove('mc-pulse'); }, 200);

    var bc = doc.querySelector('#bucket-content');
    var bl = doc.querySelector('#bucket-model-label');
    if (data[k]) {
      if (bc) bc.innerHTML = data[k].html;
      if (bl) bl.textContent = data[k].label;
    }

    syncToServer(k);
  });

  /* ── Init ─────────────────────────────────────────── */
  buildDropdown();
  adjustGridColumns();
  initDragDelegation();
})();
</script>"""


# ── Fragment: model card section ───────────────────────────
@st.fragment
def _render_model_cards(ar: dict, markets: list) -> None:
    # ── Consume URL-synced changes from JS bridge ──
    qp = st.query_params
    ml_order = qp.get("ml_order")
    ml_sel = qp.get("ml_sel")
    if ml_order:
        keys = [k.strip() for k in ml_order.split(",") if k.strip()]
        avail_set = set(_model_options(ar))
        valid = [k for k in keys if k in avail_set]
        if valid:
            st.session_state[_MODELS_KEY] = valid
    if ml_sel and ml_sel in _model_options(ar):
        st.session_state[_SELECTED_KEY] = ml_sel
    st.query_params.clear()

    sm = list(st.session_state.get(_MODELS_KEY, []))

    # ── Section title with gear icon ──
    st.markdown(
        '<div class="_mpr" style="display:flex;justify-content:space-between;'
        'align-items:center;margin-top:24px;">'
        '<div class="section-title" style="margin:0;">Model Predictions</div>'
        '<button class="_gear-btn" type="button" '
        'style="background:none;border:none;cursor:pointer;font-size:18px;'
        'color:#8F9BB7;padding:0;line-height:1;">⚙</button>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Custom dropdown panel (always rendered, hidden by default) ──
    avail = _model_options(ar)
    active_models = []
    inactive_models = []

    for mk in avail:
        label = MODEL_LABELS.get(mk, mk)
        if mk in sm:
            active_models.append({"key": mk, "label": label})
        else:
            inactive_models.append({"key": mk, "label": label})

    dropdown_html = (
        '<div id="model-dropdown-panel" style="display:none;">'
        '<div class="md-section-title">Active Models <span style="color:#6B7280;font-size:10px;">(drag to reorder)</span></div>'
        '<div id="md-active-list" class="md-list"></div>'
        '<div class="md-divider"></div>'
        '<div class="md-section-title">Inactive Models</div>'
        '<div id="md-inactive-list" class="md-list"></div>'
        '</div>'
        '<script type="application/json" id="model-dropdown-data">'
        + _json.dumps({"active": active_models, "inactive": inactive_models})
        + '</script>'
    )
    st.markdown(dropdown_html, unsafe_allow_html=True)

    # ── Sync trigger button (hidden, clicked by JS to trigger rerun) ──
    st.markdown(
        '<style>div[data-testid="stButton"]:has(button:only-child) '
        '{ position:fixed; left:-9999px; }</style>',
        unsafe_allow_html=True,
    )
    st.button("Sync", key="_js_sync_btn", on_click=lambda: None)

    # ── Pre-compute card HTML for ALL models ──
    _card_data = {}
    for mk in avail:
        pred = ar.get(mk, {})
        mv = pred.get("mean") if pred else None
        sv = pred.get("std") if pred else None
        lb = MODEL_LABELS.get(mk, mk)
        ms = f"{mv:.1f}&deg;C" if mv is not None else "--"
        ss = f"&plusmn;{sv:.2f}" if sv is not None else "&plusmn;--"
        _card_data[mk] = (
            '<div class="mc" data-key="' + mk + '">'
            '<div style="font-family:Inter,sans-serif;font-size:11px;font-weight:600;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#8F9BB7;margin-bottom:6px;">' + lb + '</div>'
            '<div style="font-family:Space Mono,monospace;font-size:32px;font-weight:700;'
            'color:#fff;line-height:1.2;margin-top:2px;">' + ms + '</div>'
            '<div style="font-family:Space Mono,monospace;font-size:11px;color:#8F9BB7;'
            'margin-top:1px;">' + ss + '</div>'
            '</div>'
        )

    # ── Pre-compute bucket data for ALL models ──
    _bucket_json = {}
    for mk in avail:
        _mk_probs = ar.get(mk, {}).get("probs", {})
        _bucket_json[mk] = {
            "html": _bucket_rows_html(markets, _mk_probs),
            "label": MODEL_LABELS.get(mk, mk),
        }

    # ── Model cards grid (ALWAYS rendered, even when empty) ──
    sel_key = st.session_state.get(_SELECTED_KEY)
    n = max(len(sm), 1)
    cards_h = '<div id="_mc-grid" style="display:grid;grid-template-columns:repeat(' + str(n) + ',1fr);gap:12px;">'
    for mk in sm:
        card_html = _card_data.get(mk, "")
        if sel_key == mk:
            card_html = card_html.replace('class="mc"', 'class="mc mc-active"', 1)
        cards_h += card_html
    cards_h += '</div>'

    # Data scripts (ALWAYS rendered, empty JSON if no data)
    cards_h += '<script type="application/json" id="card-data">' + _json.dumps(_card_data) + '</script>'
    cards_h += '<script type="application/json" id="bucket-data">' + _json.dumps(_bucket_json) + '</script>'

    st.markdown(cards_h, unsafe_allow_html=True)


def run() -> None:
    state = AppState()
    state.init_defaults()

    # ── Handle URL-synced changes from JS bridge ──
    if state.selected_event is None:
        ev = resolve_today_event()
        if ev:
            state.selected_event = ev
            from ..services.market_service import parse_date_from_event
            parsed = parse_date_from_event(ev.get("title", ""), ev.get("slug", ""))
            if parsed:
                state.target_date = parsed
            state.is_min_temp = ("lowest" in ev.get("title", "").lower() or "lowest" in ev.get("slug", "").lower())
            mark_refreshed()

    slug = state.selected_event.get("slug", "") if state.selected_event else ""
    is_min = state.is_min_temp
    td = state.target_date
    tds = pd.Timestamp(td).strftime("%Y%m%d") if td else hkt_now().strftime("%Y%m%d")
    if not slug:
        st.markdown('<div style="padding:60px 32px;max-width:1280px;margin:0 auto;"><div class="glass-card"><p style="color:#8F9BB7;">No event found.</p></div></div>', unsafe_allow_html=True)
        return

    with st.spinner("Loading data..."):
        markets = fetch_event_markets(slug, is_min)
        state.markets = markets
        hko = fetch_hko_data(tds)
        max_sf, min_sf = hko["max_since_midnight"], hko["min_since_midnight"]
        co = min_sf if is_min else max_sf
        intra = get_intraday_state(tds)
        if not intra:
            ts = hkt_now().strftime("%Y%m%d")
            if ts != tds:
                intra = get_intraday_state(ts)
        state.intraday_state = intra
        rk = {}
        if intra:
            rk = compute_rain_kwargs(tds, hkt_now())
        state.rain_kwargs = rk
        faws = hko.get("forecast_min" if is_min else "forecast_max")
        ar = run_all_models(target_date=td, target_date_str=tds, is_min_temp=is_min,
                            bias=state.bias, std_mult=state.std_mult, state=intra,
                            rain_kwargs=rk, markets=markets, forecast_aws_val=faws,
                            is_today=pd.Timestamp(td).normalize() == pd.Timestamp(hkt_now()).normalize() if td else True)
        state.pred_9d = ar.get("9d")
        state.pred_aws = ar.get("aws")
        state.pred_intra = {k: v for k, v in ar.items() if k not in ("9d", "aws", "_intraday_error")}

    # ── Warn about registered models that failed silently ──
    _INTRA_MODEL_KEYS = {k for k in MODEL_KEYS if k not in ("9d", "aws", "_intraday_error")}
    _missing = sorted(_INTRA_MODEL_KEYS - set(ar.keys()))
    if _missing:
        st.warning(
            f"The following models failed to load or predict: **{', '.join(_missing)}**. "
            "Check the Health page for details."
        )

    _init_models(ar)

    # ── Handle URL-synced changes (ml_order + ml_sel) ──
    qp = st.query_params
    ml_order = qp.get("ml_order")
    ml_sel = qp.get("ml_sel")

    if ml_order:
        keys = [k.strip() for k in ml_order.split(",") if k.strip()]
        avail = set(_model_options(ar))
        valid = [k for k in keys if k in avail]
        if valid:
            st.session_state[_MODELS_KEY] = valid

    if ml_sel:
        st.session_state[_SELECTED_KEY] = ml_sel

    st.query_params.clear()

    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    # ── Row 1: Weather cards ───────────────────────────────
    tn = intra["temp_now"] if intra and intra.get("temp_now") is not None else (co or 25.0)
    rh = intra["rh_now"] if intra and intra.get("rh_now") is not None else 70.0
    mxo = intra.get("max_so_far") if intra else None
    mno = intra.get("min_so_far") if intra else None
    cr = max(rk.get("rain_60m", 0), rk.get("rain_120m", 0)) if rk else 0.0
    r6 = rk.get("rain_60m", 0) if rk else 0.0
    r12 = rk.get("rain_120m", 0) if rk else 0.0
    rf = cr * 1.15

    w1, w2 = st.columns(2, gap="small")
    with w1:
        st.markdown(_temp_card_html(tn, rh, mxo, mno), unsafe_allow_html=True)
    with w2:
        st.markdown(_rain_card_html(cr, r6, r12, rf), unsafe_allow_html=True)

    # ── Row 2: Model cards (fragment — only this reruns on toggle) ──
    _render_model_cards(ar, markets)

    # Re-read sm from session state (may have been updated by the fragment)
    sm = list(st.session_state.get(_MODELS_KEY, []))

    # JS bridge — outside the fragment so it persists and only initializes once
    components.html(_JS_BRIDGE, height=0)

    # ── Row 3 + 4: Bucket + Chart side-by-side ────────────
    bc_col, ch_col = st.columns([1, 1], gap="small")
    with bc_col:
        if ar and markets:
            am = _active_model(sm, ar)
            ap = ar.get(am, {}).get("probs", {})
            am_label = MODEL_LABELS.get(am, am)
            rows_html = _bucket_rows_html(markets, ap)
            st.markdown(
                '<div class="section-title" style="margin-top:24px;">Probability by Bucket</div>'
                '<div class="glass-card">'
                '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
                '<div id="bucket-model-label" class="glass-label">' + am_label + '</div>'
                '<div style="display:flex;gap:14px;">'
                '<span style="display:flex;align-items:center;gap:4px;font-size:9px;color:#8F9BB7;font-family:Inter;">'
                '<span style="width:8px;height:8px;border-radius:2px;background:#A78BFA;"></span> Polymarket</span>'
                '<span style="display:flex;align-items:center;gap:4px;font-size:9px;color:#8F9BB7;font-family:Inter;">'
                '<span style="width:8px;height:8px;border-radius:2px;background:#22C55E;"></span> Model</span>'
                '</div></div>'
                '<div id="bucket-content">' + rows_html + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
    with ch_col:
        st.markdown(
            '<div class="section-title" style="margin-top:24px;">Temperature (24H)</div>'
            '<div class="glass-card" style="height:100%;">',
            unsafe_allow_html=True,
        )
        if intra and intra.get("df_today") is not None and not intra["df_today"].empty:
            st.plotly_chart(_chart(intra["df_today"]), use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<p style="color:#6B7280;font-size:12px;padding:20px 0;">No intraday data available.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
