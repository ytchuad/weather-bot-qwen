from __future__ import annotations

import json as _json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..components.top_nav import mark_refreshed
from ..config import MODEL_LABELS, TMAX_BUCKETS, TMIN_BUCKETS
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
_DEFAULT_MODELS = ["baseline", "rain_nowcast", "model_a"]


def _init_models(all_results: dict):
    if _MODELS_KEY not in st.session_state:
        avail = [k for k in all_results if k not in ("9d", "aws", "_intraday_error")]
        st.session_state[_MODELS_KEY] = [m for m in _DEFAULT_MODELS if m in avail][:3]
    if _SELECTED_KEY not in st.session_state:
        sm = st.session_state.get(_MODELS_KEY, [])
        st.session_state[_SELECTED_KEY] = sm[0] if sm else None


def _model_options(all_results: dict) -> list[str]:
    return [k for k in all_results if k not in ("9d", "aws", "_intraday_error")]


def _active_model(selected_models: list[str], all_results: dict) -> str:
    m = st.session_state.get(_SELECTED_KEY)
    if m and m in all_results:
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
(function(){
  function wire(){
    var bd = document.querySelector('#bucket-data');
    var bucketContent = document.querySelector('#bucket-content');
    if (!bd || !bucketContent) { setTimeout(wire, 200); return; }

    /* Guard: only add the click listener once (survives fragment & full reruns) */
    if (document.body.hasAttribute('data-mc-wired')) return;
    document.body.setAttribute('data-mc-wired', '1');

    /* Hide any visible Streamlit text inputs on this page
       (they are bridge widgets that should not be user-visible) */
    try {
      var allDivs = document.querySelectorAll('div[data-testid="stElementContainer"]');
      allDivs.forEach(function(div){
        /* If container has no visible content except a label, hide it */
        var vis = div.querySelector('input');
        var txt = div.querySelector('textarea');
        if ((vis && vis.type === 'text' && !vis.placeholder) || txt) {
          var label = div.querySelector('label') || div.querySelector('[data-baseweb="label"]');
          var isBridge = label && (label.textContent.trim().charAt(0) === '_');
          if (isBridge) div.style.display = 'none';
        }
      });
    } catch(e){}

    document.body.addEventListener('click', function(e){
      var card = e.target.closest('.mc');
      if (!card) return;
      var k = card.getAttribute('data-key');
      if (!k) return;

      /* Re-read bucket data each click (DOM may have been rebuilt by fragment rerun) */
      var freshBd = document.querySelector('#bucket-data');
      var data = freshBd ? JSON.parse(freshBd.textContent) : {};

      /* 1. Update active card styling */
      document.querySelectorAll('.mc').forEach(function(c){
        c.classList.remove('mc-active');
        c.classList.remove('mc-pulse');
      });
      card.classList.add('mc-active');
      card.classList.add('mc-pulse');
      setTimeout(function(){ card.classList.remove('mc-pulse'); }, 200);

      /* 2. Update bucket bars — instant, no rerun */
      var bc = document.querySelector('#bucket-content');
      var bl = document.querySelector('#bucket-model-label');
      if (data[k]) {
        if (bc) bc.innerHTML = data[k].html;
        if (bl) bl.textContent = data[k].label;
      }

    });
  }
  wire();
})();
</script>"""


# ── Fragment: model card section ───────────────────────────
# Only this section reruns when a toggle changes — the rest of
# the page (weather cards, bucket bars, chart) stays untouched.
@st.fragment
def _render_model_cards(ar: dict, markets: list) -> None:
    sm = list(st.session_state.get(_MODELS_KEY, []))

    # ── Section title + gear popover ──────────────────────────
    cols = st.columns([10, 1])
    with cols[0]:
        st.markdown(
            '<div class="section-title" id="mp-title" style="margin-top:24px;">Model Predictions</div>',
            unsafe_allow_html=True,
        )
    _selected_removed = False
    with cols[1]:
        with st.popover("⚙", use_container_width=False):
            avail = _model_options(ar)
            cur = set(sm)
            for m in avail:
                is_on = m in cur
                label = MODEL_LABELS.get(m, m)
                new_val = st.toggle(
                    f"{label}  [{m}]", value=is_on, key=f"mtg_{m}"
                )
                if new_val != is_on:
                    if new_val:
                        sm.append(m)
                    else:
                        sm.remove(m)
                    st.session_state[_MODELS_KEY] = list(sm)
                    cur_sel = st.session_state.get(_SELECTED_KEY)
                    if cur_sel not in sm:
                        st.session_state[_SELECTED_KEY] = sm[0] if sm else None
                        _selected_removed = True

    # ── Model cards ──────────────────────────────────────────
    if sm:
        sel_key = st.session_state.get(_SELECTED_KEY)

        # Glass cards — rendered via st.markdown so they land in the main DOM
        n = len(sm)
        cards_h = '<div style="display:grid;grid-template-columns:repeat(' + str(n) + ',1fr);gap:12px;">'
        for mk in sm:
            pred = ar.get(mk, {})
            mv = pred.get("mean") if pred else None
            sv = pred.get("std") if pred else None
            lb = MODEL_LABELS.get(mk, mk)
            ms = f"{mv:.1f}&deg;C" if mv is not None else "--"
            ss = f"&plusmn;{sv:.2f}" if sv is not None else "&plusmn;--"
            is_sel = sel_key == mk
            active_cls = " mc-active" if is_sel else ""
            cards_h += (
                '<div class="mc' + active_cls + '" data-key="' + mk + '">'
                '<div style="font-family:Inter,sans-serif;font-size:11px;font-weight:600;letter-spacing:0.1em;'
                'text-transform:uppercase;color:#8F9BB7;margin-bottom:6px;">' + lb + '</div>'
                '<div style="font-family:Space Mono,monospace;font-size:32px;font-weight:700;'
                'color:#fff;line-height:1.2;margin-top:2px;">' + ms + '</div>'
                '<div style="font-family:Space Mono,monospace;font-size:11px;color:#8F9BB7;'
                'margin-top:1px;">' + ss + '</div>'
                '</div>'
            )
        cards_h += '</div>'

        # Pre-render bucket data for ALL selected models (consumed by JS bridge)
        _bucket_json = {}
        for mk in sm:
            _mk_probs = ar.get(mk, {}).get("probs", {})
            _bucket_json[mk] = {
                "html": _bucket_rows_html(markets, _mk_probs),
                "label": MODEL_LABELS.get(mk, mk),
            }
        cards_h += '<script type="application/json" id="bucket-data">' + _json.dumps(_bucket_json) + '</script>'
        st.markdown(cards_h, unsafe_allow_html=True)

    # If the selected model was toggled off, the bucket section (outside this
    # fragment) needs a full rerun to show the fallback model's data.
    if _selected_removed:
        st.rerun()


def run() -> None:
    state = AppState()
    state.init_defaults()

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

    _init_models(ar)
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
    st.html(_JS_BRIDGE, unsafe_allow_javascript=True)

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
