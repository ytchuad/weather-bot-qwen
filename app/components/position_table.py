# app/components/position_table.py
"""Position detail table with per-position sell buttons."""

import streamlit as st

from ..services.strategy_service import sell_position


def position_table(
    details: list[dict],
    portfolio_id: str,
    slug: str = "",
) -> None:
    """Render position detail table with sell buttons.

    Args:
        details: list of position dicts from get_pnl()['details'].
        portfolio_id: current portfolio ID.
        slug: current event slug.
    """
    if not details:
        st.info("No positions.")
        return

    headers = ["Strategy", "Bucket", "Side", "Qty", "Entry", "Market", "Cost", "Mkt Val", "Fee", "Action"]
    header_cols = st.columns([2, 2, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1])
    for hc, hdr in zip(header_cols, headers):
        hc.markdown(f"**{hdr}**")

    for idx, d in enumerate(details):
        # Summary rows (subtotal lines)
        if d.get("_summary"):
            cols = st.columns([2, 8, 4])
            cols[0].markdown(f"**{d['slug']}**")
            cols[1].markdown(
                f"**(Subtotal)** Cost=${d['cost_basis']:,.2f}  "
                f"MktVal=${d['market_value']:,.2f}  PnL=${d['pnl']:+,.2f}"
            )
            continue

        side_label = d.get("side", "YES")
        fee_str = f"${d.get('fee', 0.0):,.2f}" if d.get("fee", 0) > 0 else "—"

        cols = st.columns([2, 2, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1])
        cols[0].write(d.get("strategy", ""))
        cols[1].write(d.get("bucket", ""))
        cols[2].write(side_label)
        cols[3].write(str(d.get("quantity", 0)))
        cols[4].write(f"{d.get('entry_price', 0) * 100:.1f}¢")
        cols[5].write(f"{d.get('current_price', 0) * 100:.1f}¢")
        cols[6].write(f"${d.get('cost_basis', 0):,.2f}")
        cols[7].write(f"${d.get('market_value', 0):,.2f}")
        cols[8].write(fee_str)

        _sell_key = f"sell_{portfolio_id}_{d.get('slug', slug)}_{d.get('bucket', '')}_{side_label}_{idx}"
        if cols[9].button("Sell", key=_sell_key, type="secondary"):
            try:
                outcome = "yes" if side_label == "YES" else "no"
                result = sell_position(d["bucket"], outcome, float(d["quantity"]))
                if result:
                    st.success(
                        f"Sold {d['bucket']} {result['sold_qty']:.2f}sh  "
                        f"proceeds=${result['proceeds']:.2f}  fee=${result['fee']:.4f}"
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Sell failed: {e}")
