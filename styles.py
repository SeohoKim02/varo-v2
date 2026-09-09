"""Shared visual system for Varo V2."""
from __future__ import annotations

import streamlit as st

DESIGN_TOKENS = {
    "app_bg": "#FAFBFC",
    "card_bg": "#FFFFFF",
    "panel_soft": "#F3F5F7",
    "text": "#1F2937",
    "strong_text": "#111827",
    "muted_text": "#4B5563",
    "border": "#E5E7EB",
    "accent": "#2d6fa8",
    "accent_soft": "#EAF3FF",
    "accent_border": "#CFE4FB",
    "success": "#1f8a5b",
    "warning": "#b7791f",
    "error": "#c2412d",
    "info": "#2d5f9a",
    # Two corner sizes and one pill, nothing else. Cards/tables/network use the
    # medium radius, controls and list rows the small one; 999px is for badges.
    "card_radius": "10px",
    "button_radius": "6px",
    # Hierarchy comes from border + spacing, so the shadow is a hairline used on
    # the one floating surface (the top bar), never on every card.
    "shadow": "0 1px 2px rgba(17, 24, 39, 0.04)",
}

# One type scale for the whole product. Every visible role below maps to exactly
# one size/weight pair, so two screens can never disagree about what a "section
# title" looks like. Sizes are rem so the browser's own text scaling still works.
#
# The dense-network SVG is deliberately absent: its label sizes are measured
# against each node's box by components/workspace_network.py and must stay on
# their own rules (see .ws-network-svg below).
TYPE_SCALE = {
    "page_size": "1.5rem",        # 24px — the one page title, all four screens
    "page_weight": "750",
    "section_size": "1.0625rem",  # 17px — 결론 / 상태 / 재고 이동 네트워크 …
    "section_weight": "750",
    "card_size": "0.9375rem",     # 15px — card and in-card block titles
    "card_weight": "700",
    "metric_size": "1.75rem",     # 28px — KPI values and st.metric alike
    "metric_weight": "800",
    # 17px is also where a card's *answer* is written — the route of the selected
    # move, the applied data state, a verification verdict. One size, two weights:
    # 800 for the single headline value of a card, 700 for the supporting ones.
    # Without these two names the same 17px kept being re-typed at 700/750/800 and
    # a value became indistinguishable from a section title.
    "lead_size": "1.0625rem",     # 17px — the answer a card exists to give
    "lead_weight": "800",
    "value_weight": "700",
    "body_size": "0.875rem",      # 14px — default readable body
    "secondary_size": "0.8125rem",  # 13px — supporting lines
    "caption_size": "0.75rem",    # 12px — captions, legends, badges
}

# Vertical rhythm: two gaps and one card padding, reused everywhere.
SPACING = {
    "section_gap": "1.05rem",
    "block_gap": "0.55rem",
    "card_padding": "0.95rem 1rem",
}

# Streamlit renders ``st.container(key=X)`` with a ``st-key-X`` class. The 재고 운영
# three-column row carries this key so the narrow-desktop rules below can find it;
# pages/workspace.py imports the same constant, so the hook cannot drift.
WORKSPACE_MAIN_ROW_KEY = "ws_main_row"


def apply_global_styles() -> None:
    """Apply scoped V2 styles."""
    main_row = WORKSPACE_MAIN_ROW_KEY
    type_scale = TYPE_SCALE
    spacing = SPACING
    st.markdown(
        f"""
        <style>
        :root {{
            --varo-bg: {DESIGN_TOKENS['app_bg']};
            --varo-panel: {DESIGN_TOKENS['card_bg']};
            --varo-panel-soft: {DESIGN_TOKENS['panel_soft']};
            --varo-text: {DESIGN_TOKENS['text']};
            --varo-strong: {DESIGN_TOKENS['strong_text']};
            --varo-muted: {DESIGN_TOKENS['muted_text']};
            --varo-line: {DESIGN_TOKENS['border']};
            --varo-accent: {DESIGN_TOKENS['accent']};
            --varo-accent-soft: {DESIGN_TOKENS['accent_soft']};
            --varo-accent-border: {DESIGN_TOKENS['accent_border']};
            --varo-success: {DESIGN_TOKENS['success']};
            --varo-warning: {DESIGN_TOKENS['warning']};
            --varo-error: {DESIGN_TOKENS['error']};
            --varo-info: {DESIGN_TOKENS['info']};
            --varo-radius-card: {DESIGN_TOKENS['card_radius']};
            --varo-radius-button: {DESIGN_TOKENS['button_radius']};
            --varo-shadow: {DESIGN_TOKENS['shadow']};
            --varo-fs-page: {type_scale['page_size']};
            --varo-fw-page: {type_scale['page_weight']};
            --varo-fs-section: {type_scale['section_size']};
            --varo-fw-section: {type_scale['section_weight']};
            --varo-fs-card: {type_scale['card_size']};
            --varo-fw-card: {type_scale['card_weight']};
            --varo-fs-metric: {type_scale['metric_size']};
            --varo-fw-metric: {type_scale['metric_weight']};
            --varo-fs-lead: {type_scale['lead_size']};
            --varo-fw-lead: {type_scale['lead_weight']};
            --varo-fw-value: {type_scale['value_weight']};
            --varo-fs-body: {type_scale['body_size']};
            --varo-fs-secondary: {type_scale['secondary_size']};
            --varo-fs-caption: {type_scale['caption_size']};
            --varo-section-gap: {spacing['section_gap']};
            --varo-block-gap: {spacing['block_gap']};
            --varo-card-padding: {spacing['card_padding']};
        }}
        /* Force the app onto a light surface regardless of OS/browser dark mode. */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
            background: var(--varo-bg) !important;
            color: var(--varo-text);
        }}
        [data-testid="stHeader"], header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
        [data-testid="stToolbar"] {{ color: var(--varo-muted); }}
        .stApp {{
            background: var(--varo-bg);
            color: var(--varo-text);
        }}
        .block-container {{
            /* The result has to start near the top of a 768px-tall desktop, so the
               top padding only has to clear the Streamlit header strip. */
            padding-top: 1.5rem !important;
            padding-bottom: 3rem;
            max-width: 1480px;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        /* On a wide desktop the network is the part that benefits from the space
           the 1480px cap was leaving unused on both sides. */
        /* Below ~1560px the 5rem side gutters cost more than they are worth: the
           workspace has three columns to fit, and the centre network is the part
           that stops being legible first. Wider screens are capped by max-width
           anyway, so this changes nothing above the breakpoint. */
        @media (min-width: 641px) and (max-width: 1560px) {{
            .block-container {{ padding-left: 2rem !important; padding-right: 2rem !important; }}
        }}
        @media (min-width: 1700px) {{
            .block-container {{ max-width: 1660px; }}
        }}
        .v2-wrap, .v2-wrap * {{
            box-sizing: border-box;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .v2-topbar {{
            display: grid;
            grid-template-columns: minmax(210px, 1fr) auto;
            gap: 0.7rem;
            align-items: center;
            border: 1px solid var(--varo-line);
            background: var(--varo-panel);
            border-radius: var(--varo-radius-card);
            box-shadow: var(--varo-shadow);
            padding: 0.68rem 0.9rem;
            margin-bottom: 0.48rem;
        }}
        .v2-brand {{
            font-size: var(--varo-fs-lead);
            font-weight: var(--varo-fw-lead);
            letter-spacing: 0.01em;
            color: var(--varo-text);
        }}
        .v2-topbar-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            justify-content: flex-end;
            align-items: center;
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
        }}
        .v2-pill, .v2-file-label {{
            border: 1px solid var(--varo-line);
            border-radius: 999px;
            padding: 0.22rem 0.56rem;
            background: #f8fafc;
        }}
        .v2-pill {{
            color: var(--varo-accent);
            background: var(--varo-accent-soft);
            border-color: var(--varo-accent-border);
            font-weight: 700;
        }}
        .v2-data-onboarding {{
            border: 1px solid var(--varo-accent-border);
            background: var(--varo-accent-soft);
            border-radius: var(--varo-radius-card);
            padding: 0.62rem 0.78rem;
            margin: 0.25rem 0 0.4rem;
        }}
        .v2-data-title {{
            color: var(--varo-text);
            font-size: var(--varo-fs-card);
            font-weight: var(--varo-fw-card);
            margin-bottom: 0.14rem;
        }}
        .v2-data-bar-compact {{
            min-height: 38px;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            border: 1px solid var(--varo-line);
            background: var(--varo-panel);
            border-radius: var(--varo-radius-button);
            padding: 0.34rem 0.5rem;
        }}
        .v2-data-filename {{
            min-width: 0;
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
            overflow-wrap: anywhere;
        }}
        .v2-page-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.8rem;
            margin: 0.48rem 0 0.62rem;
        }}
        /* The title is an <h1>, and Streamlit's own markdown h1 rule is more
           specific than a bare class, so it used to win and paint this at 44px
           while the workspace title sat at 18px. Matching the element as well as
           the class puts all four screens on one page-title size. */
        .v2-page-title,
        .v2-wrap h1.v2-page-title {{
            font-size: var(--varo-fs-page);
            font-weight: var(--varo-fw-page);
            line-height: 1.25;
            color: var(--varo-text);
            margin: 0;
            padding: 0;
            letter-spacing: -0.01em;
        }}
        .v2-page-desc {{
            margin-top: 0.22rem;
            color: var(--varo-muted);
            font-size: var(--varo-fs-body);
            line-height: 1.45;
        }}
        /* Cards carry no shadow: a 1px border plus the section gap already
           separates them from the page, and shadowing every card was what made
           the screen read as a dashboard template. */
        .v2-card {{
            border: 1px solid var(--varo-line);
            background: var(--varo-panel);
            border-radius: var(--varo-radius-card);
            box-shadow: none;
            padding: var(--varo-card-padding);
            min-width: 0;
        }}
        .v2-card-head {{
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            align-items: center;
            margin-bottom: 0.65rem;
        }}
        .v2-kpi-card {{
            padding: 0.85rem 0.9rem;
            min-height: 128px;
        }}
        .v2-kpi-card-compact {{
            min-height: 132px;
            padding: 0.85rem 0.92rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        .v2-kpi-card-compact .v2-kpi-value {{
            font-size: var(--varo-fs-metric);
            font-weight: var(--varo-fw-metric);
            margin-top: 0.18rem;
        }}
        .v2-kpi-title {{
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
            font-weight: 700;
        }}
        .v2-kpi-desc {{
            color: var(--varo-muted);
            font-size: var(--varo-fs-caption);
            line-height: 1.4;
            margin-top: 0.4rem;
        }}
        .v2-card-title {{
            color: var(--varo-text);
            font-weight: var(--varo-fw-card);
            font-size: var(--varo-fs-card);
            margin-bottom: 0.4rem;
        }}
        .v2-card-caption {{
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
            line-height: 1.45;
        }}
        .v2-kpi-value {{
            font-size: var(--varo-fs-metric);
            font-weight: var(--varo-fw-metric);
            line-height: 1.12;
            color: var(--varo-text);
            margin-top: 0.12rem;
            min-width: 0;
            overflow-wrap: anywhere;
            word-break: keep-all;
        }}
        .v2-section-header {{
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            margin: var(--varo-section-gap) 0 var(--varo-block-gap);
        }}
        .v2-section-title {{
            font-size: var(--varo-fs-section);
            font-weight: var(--varo-fw-section);
            color: var(--varo-text);
        }}
        .v2-section-desc {{
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
            margin-top: 0.1rem;
        }}
        /* Every empty state (no data, before analysis, no plan, no filter match)
           is the same quiet dashed panel: a short title, one explanation line,
           and whatever the next action is, placed underneath by the caller. */
        .v2-empty-state, .v2-error-card {{
            border: 1px dashed var(--varo-line);
            background: #fbfcfd;
            border-radius: var(--varo-radius-card);
            padding: 0.92rem 1rem;
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
            line-height: 1.5;
            min-height: 68px;
        }}
        .v2-empty-state strong, .v2-error-card strong {{
            display: block;
            font-size: var(--varo-fs-card);
            font-weight: var(--varo-fw-card);
            color: var(--varo-text);
        }}
        .v2-error-card strong {{ color: var(--varo-error); }}
        .v2-empty-state-compact {{ min-height: 0; padding: 0.72rem 0.82rem; }}
        /* Home status card: roomy, never fixed-height so Korean text/buttons never clip. */
        .v2-home-state-card {{
            padding: 1.15rem 1.25rem;
            min-height: 0;
        }}
        /* This card carries the next step for a whole screen, so its heading is a
           section heading and takes that level exactly — at 17px/700 it was a
           third weight on a size the app already uses for two other roles. */
        .v2-home-state-card .v2-card-title {{
            font-size: var(--varo-fs-section);
            font-weight: var(--varo-fw-section);
            line-height: 1.4;
        }}
        .v2-home-state-card .v2-card-caption {{ line-height: 1.6; white-space: normal; word-break: keep-all; }}
        .v2-error-card {{
            border-color: rgba(194, 65, 45, 0.35);
            background: #fff8f6;
            color: var(--varo-error);
        }}
        .v2-badge {{
            display: inline-flex;
            align-items: center;
            max-width: 100%;
            border-radius: 999px;
            padding: 0.22rem 0.54rem;
            font-size: var(--varo-fs-caption);
            font-weight: 700;
            line-height: 1.2;
            border: 1px solid var(--varo-line);
            white-space: normal;
        }}
        .v2-badge-neutral {{ background: #f3f5f7; color: var(--varo-muted); }}
        .v2-badge-accent {{ background: var(--varo-accent-soft); color: var(--varo-accent); border-color: var(--varo-accent-border); }}
        .v2-badge-success {{ background: #e8f7ef; color: var(--varo-success); border-color: #c9ead8; }}
        .v2-badge-warning {{ background: #fff7dc; color: #8a6400; border-color: #f1db8a; }}
        .v2-badge-error {{ background: #fff0ed; color: var(--varo-error); border-color: #f3c3ba; }}
        .v2-detail-row {{
            display: grid;
            grid-template-columns: 112px minmax(0, 1fr);
            gap: 0.6rem;
            padding: 0.36rem 0;
            border-bottom: 1px solid var(--varo-line);
            align-items: start;
        }}
        .v2-detail-row:last-child {{ border-bottom: 0; }}
        .v2-recommendation-info {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0;
            margin-top: 0.55rem;
            padding: 0.78rem 0.85rem;
        }}
        .v2-info-item {{
            min-width: 0;
            padding: 0 0.85rem;
            border-right: 1px solid var(--varo-line);
        }}
        .v2-info-item:first-child {{ padding-left: 0; }}
        .v2-info-item:last-child {{ padding-right: 0; border-right: 0; }}
        /* Three columns for the six-fact applied-data card: two even rows
           instead of a half-empty second row. */
        .v2-info-grid-3 {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
            row-gap: 0.85rem;
        }}
        .v2-info-grid-3 .v2-info-item:nth-child(3n) {{ border-right: 0; padding-right: 0; }}
        .v2-info-grid-3 .v2-info-item:nth-child(3n + 1) {{ padding-left: 0; }}
        .v2-info-item strong {{
            display: block;
            margin-top: 0.2rem;
            color: var(--varo-text);
            font-size: var(--varo-fs-body);
            line-height: 1.4;
            white-space: normal;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .v2-table-shell {{
            border: 1px solid var(--varo-line);
            background: var(--varo-panel);
            border-radius: var(--varo-radius-card);
            overflow: visible;
        }}
        .v2-html-table-wrap {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--varo-line);
            border-radius: var(--varo-radius-card);
            background: var(--varo-panel);
        }}
        .v2-html-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: var(--varo-fs-secondary);
            color: var(--varo-text);
        }}
        .v2-html-table thead th {{
            background: var(--varo-panel-soft);
            color: var(--varo-muted);
            font-weight: 700;
            text-align: left;
            padding: 0.52rem 0.66rem;
            border-bottom: 1px solid var(--varo-line);
            white-space: nowrap;
        }}
        .v2-html-table tbody td {{
            padding: 0.5rem 0.66rem;
            border-bottom: 1px solid var(--varo-line);
            color: var(--varo-text);
            white-space: nowrap;
        }}
        /* Numbers line up on their last digit, so a column of quantities or
           amounts can be compared by eye. components/tables.py tags the cells. */
        .v2-html-table th.v2-num, .v2-html-table td.v2-num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .v2-html-table tbody tr:nth-child(2n) td {{ background: #fcfdfe; }}
        .v2-html-table tbody tr:last-child td {{ border-bottom: 0; }}
        .v2-html-table tbody tr.v2-row-pick td {{ background: var(--varo-accent-soft); font-weight: 700; }}
        /* The virtualised grid gets the same numeric treatment it can accept. */
        [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
        .v2-hbar {{ display: flex; flex-direction: column; gap: 0.5rem; margin: 0.3rem 0 0.2rem; }}
        .v2-hbar-row {{ display: grid; grid-template-columns: 120px 1fr 56px; gap: 0.6rem; align-items: center; }}
        .v2-hbar-label {{ color: var(--varo-text); font-size: var(--varo-fs-body); font-weight: 600; }}
        .v2-hbar-track {{ background: var(--varo-panel-soft); border: 1px solid var(--varo-line); border-radius: 999px; height: 16px; overflow: hidden; }}
        .v2-hbar-fill {{ display: block; height: 100%; background: var(--varo-accent-soft); border-right: 2px solid var(--varo-accent); }}
        .v2-hbar-value {{ color: var(--varo-muted); font-size: var(--varo-fs-body); text-align: right; font-variant-numeric: tabular-nums; }}
        @media (max-width: 640px) {{ .v2-hbar-row {{ grid-template-columns: 90px 1fr 44px; }} }}
        .v2-network-shell {{
            position: relative;
            width: 100%;
            min-height: 620px;
            border: 1px solid var(--varo-line);
            border-radius: var(--varo-radius-card);
            background: #f8fafb;
            overflow: hidden;
        }}
        .v2-network-placeholder {{
            min-height: 620px;
            display: grid;
            place-items: center;
            color: var(--varo-muted);
            text-align: center;
            padding: 1rem;
        }}
        .v2-network-svg {{ display: block; width: 100%; height: 620px; }}
        .v2-network-svg text {{ font-family: inherit; fill: var(--varo-text); }}
        .v2-network-svg .node-label {{ font-size: 15.5px; font-weight: 780; }}
        .v2-network-svg .dc-label {{ font-size: 17.5px; font-weight: 820; }}
        .v2-network-svg .node-type {{ font-size: 11px; fill: var(--varo-muted); }}
        .v2-network-svg .store-sub {{ font-size: 11px; fill: var(--varo-muted); }}
        .v2-network-svg .network-node {{ filter: drop-shadow(0 2px 3px rgba(30, 41, 59, 0.06)); }}
        .v2-network-svg .recommended-node {{ filter: drop-shadow(0 3px 5px rgba(216, 131, 120, 0.18)); }}
        .v2-network-svg .v2-vehicle {{ filter: drop-shadow(0 3px 5px rgba(30, 41, 59, 0.20)); }}
        .v2-network-svg .vehicle-route {{ font-size: 10.2px; font-weight: 840; }}
        .v2-network-svg .vehicle-mode {{ font-size: 8.8px; font-weight: 760; }}
        .v2-network-svg .vehicle-type {{ font-size: 8px; font-weight: 800; fill: #ffffff; }}
        .v2-network-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            padding: 0.52rem 0.65rem 0;
            color: var(--varo-muted);
            font-size: var(--varo-fs-caption);
        }}
        .v2-legend-line {{ width: 28px; height: 0; border-top: 2px solid var(--varo-accent); }}
        .v2-legend-line-dashed {{ border-top-style: dashed; }}
        .v2-legend-state {{ display: inline-flex; align-items: center; gap: 0.22rem; }}
        .v2-legend-dot {{ width: 9px; height: 9px; border-radius: 50%; border: 1px solid; display: inline-block; }}
        .v2-running-route {{
            border: 1px solid var(--varo-line);
            border-radius: var(--varo-radius-button);
            padding: 0.5rem 0.55rem;
            margin-top: 0.42rem;
            background: #fbfcfd;
        }}
        .v2-running-route-selected {{ border: 2px solid var(--varo-accent); }}
        .v2-running-route strong {{ display: block; font-size: var(--varo-fs-body); line-height: 1.25; }}
        .v2-running-route-meta {{
            display: grid;
            gap: 0.32rem;
            margin-top: 0.28rem;
            color: var(--varo-muted);
            font-size: var(--varo-fs-caption);
        }}
        .v2-route-code {{ color: var(--varo-muted); font-size: var(--varo-fs-caption); margin-left: 0.25rem; }}
        /* Active menu / primary action: soft light-blue accent, dark readable text. */
        .stButton button[kind="primary"],
        button[data-testid="stBaseButton-primary"],
        button[data-testid="baseButton-primary"] {{
            background-color: var(--varo-accent-soft) !important;
            border: 1px solid var(--varo-accent-border) !important;
            color: #1e4f7a !important;
            font-weight: 720 !important;
            box-shadow: none !important;
        }}
        .stButton button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {{
            background-color: #DCEBFF !important;
            border-color: #a9cdf0 !important;
            color: #163f63 !important;
        }}
        /* ---- Streamlit native widgets → light surfaces ---- */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            background: var(--varo-panel) !important;
            border-right: 1px solid var(--varo-line);
        }}
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ color: var(--varo-text); }}
        /* Secondary / default buttons stay white with dark text. */
        .stButton button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="baseButton-secondary"] {{
            background-color: var(--varo-panel) !important;
            color: var(--varo-text) !important;
            border: 1px solid var(--varo-line) !important;
            box-shadow: none !important;
        }}
        .stButton button[kind="secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {{
            background-color: var(--varo-panel-soft) !important;
            border-color: #cfd6de !important;
            color: var(--varo-text) !important;
        }}
        .stButton button:disabled {{ opacity: 0.5 !important; }}
        /* One button size and one label style for the whole app, so a control's
           role is read from its fill, not from an accidental size difference. */
        .stButton button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {{
            border-radius: var(--varo-radius-button) !important;
            min-height: 38px;
            padding: 0.32rem 0.85rem;
        }}
        .stButton button p, [data-testid="stDownloadButton"] button p {{
            font-size: var(--varo-fs-body) !important;
            font-weight: 700;
        }}
        /* Secondary and download buttons are the same tier and must look it. */
        [data-testid="stDownloadButton"] button {{ font-weight: 700; }}
        /* ---- Alerts -------------------------------------------------------- */
        /* Streamlit's alerts are large tinted slabs. Kept as the same components
           (no DOM hacks) but sized to the product: a normal, healthy state should
           not shout, and a stale result must read as a notice rather than a fault. */
        /* The outer stAlert is only a wrapper: giving it a border too drew a
           frame around the tinted panel inside it. The tint carries the border. */
        [data-testid="stAlert"] {{
            border: 0;
            padding: 0;
            background: transparent;
        }}
        [data-testid="stAlert"] > div {{
            border-radius: var(--varo-radius-card);
            padding: 0.62rem 0.85rem;
            border: 1px solid var(--varo-line);
        }}
        [data-testid="stAlert"] p {{
            font-size: var(--varo-fs-secondary);
            line-height: 1.5;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        [data-testid="stAlertContentSuccess"] {{
            background: #f2faf5 !important;
            border-color: #cde9d9 !important;
        }}
        [data-testid="stAlertContentWarning"] {{
            background: #fffaf0 !important;
            border-color: #f0e0b4 !important;
        }}
        [data-testid="stAlertContentInfo"] {{
            background: #f6f9fd !important;
            border-color: var(--varo-accent-border) !important;
        }}
        [data-testid="stAlertContentError"] {{
            background: #fff7f5 !important;
            border-color: #f2c8bf !important;
        }}
        /* Download buttons */
        [data-testid="stDownloadButton"] button {{
            background-color: var(--varo-panel) !important;
            color: var(--varo-text) !important;
            border: 1px solid var(--varo-line) !important;
        }}
        /* File uploader dropzone */
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--varo-panel-soft) !important;
            border: 1px dashed var(--varo-line) !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{ color: var(--varo-text) !important; }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: var(--varo-panel) !important;
            color: var(--varo-text) !important;
            border: 1px solid var(--varo-line) !important;
        }}
        /* Inputs / select / number. BaseWeb ships its own 8px corner, which was
           a third radius sitting next to the app's 10px cards and 6px controls. */
        [data-baseweb="select"] > div, [data-baseweb="input"] {{
            background-color: var(--varo-panel) !important;
            border-color: var(--varo-line) !important;
            border-radius: var(--varo-radius-button) !important;
        }}
        [data-baseweb="select"] > div > div, [data-baseweb="input"] > div {{
            border-radius: var(--varo-radius-button) !important;
        }}
        .stTextInput input, .stNumberInput input, [data-baseweb="input"] input,
        [data-baseweb="select"] input, textarea {{
            background-color: var(--varo-panel) !important;
            color: var(--varo-text) !important;
        }}
        [data-baseweb="popover"] div[role="listbox"], [data-baseweb="menu"] {{
            background: var(--varo-panel) !important;
        }}
        /* Tabs: selected tab gets a soft blue chip, dark text. The label sits on
           the body size so a tab row never competes with the section title above
           it — Streamlit's own default runs a step larger. */
        .stTabs [data-baseweb="tab"] {{ color: var(--varo-muted) !important; background: transparent; }}
        .stTabs [data-baseweb="tab"] p {{
            font-size: var(--varo-fs-body) !important;
            font-weight: 700;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: var(--varo-accent) !important;
            background: var(--varo-accent-soft) !important;
            border-radius: var(--varo-radius-button) var(--varo-radius-button) 0 0;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--varo-accent) !important; }}
        .stTabs [data-baseweb="tab-border"] {{ background-color: var(--varo-line) !important; }}
        /* Expander */
        [data-testid="stExpander"] {{
            background: var(--varo-panel);
            border: 1px solid var(--varo-line) !important;
            border-radius: var(--varo-radius-card);
            box-shadow: none !important;
        }}
        [data-testid="stExpander"] details {{
            box-shadow: none !important;
            border-radius: var(--varo-radius-card) !important;
        }}
        [data-testid="stExpander"] summary {{
            border-radius: var(--varo-radius-card) var(--varo-radius-card) 0 0 !important;
        }}
        [data-testid="stExpander"] details:not([open]) summary {{
            border-radius: var(--varo-radius-card) !important;
        }}
        [data-testid="stExpander"] summary {{
            background: var(--varo-panel-soft);
            color: var(--varo-text);
            font-size: var(--varo-fs-body);
        }}
        /* Streamlit ships its own drop shadow on bordered containers and popovers
           (rgba(0,0,0,.08) 1px 2px 8px). Two shadow languages on one screen read
           as two products, so the app keeps only the hairline defined above. */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stPopoverBody"],
        [data-testid="stElementToolbar"],
        [data-testid="stElementToolbarButtonContainer"],
        [data-testid="stDataFrame"] {{ box-shadow: none !important; }}
        /* The grid's hover toolbar is the last surface that still drew Streamlit's
           own 8px corner and drop shadow; it floats over a table on four screens,
           so it read as a control from a different app. */
        [data-testid="stElementToolbarButtonContainer"],
        [data-testid="stBaseButton-elementToolbar"] {{
            border-radius: var(--varo-radius-button) !important;
        }}
        [data-testid="stExpander"] summary:hover {{ color: var(--varo-accent); }}
        /* Metric — let long Korean status text wrap instead of being clipped, and
           put it on the product KPI scale. Streamlit's default paints the value
           at 36px/400, which made the same "KPI" idea look like a different
           product on the validation screen than the 28px/800 cards elsewhere. */
        [data-testid="stMetric"] {{
            background: var(--varo-panel);
            border: 1px solid var(--varo-line);
            border-radius: var(--varo-radius-card);
            padding: var(--varo-card-padding);
            color: var(--varo-text);
            min-width: 0;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--varo-text);
            font-size: var(--varo-fs-metric) !important;
            font-weight: var(--varo-fw-metric) !important;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: keep-all;
            line-height: 1.2;
            margin-top: 0.2rem;
        }}
        [data-testid="stMetricValue"] > div {{
            white-space: normal;
            overflow-wrap: anywhere;
            font-size: inherit;
            font-weight: inherit;
        }}
        [data-testid="stMetricLabel"] {{ color: var(--varo-muted); }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {{
            font-size: var(--varo-fs-secondary) !important;
            font-weight: 700;
        }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] p {{ white-space: normal; overflow-wrap: anywhere; word-break: keep-all; }}
        /* The verdict row on the validation screen states the same four checks the
           operations screen already prints at the lead level. A word is not a KPI
           number: at 28px/800 the two screens showed one fact in two languages, so
           this row is pinned to the same level as its twin. The widget stays an
           st.metric — the integration test reads it — only its size moves. */
        .st-key-validation_conclusion [data-testid="stMetricValue"] {{
            font-size: var(--varo-fs-lead) !important;
            font-weight: var(--varo-fw-value) !important;
        }}
        /* Dataframe wrapper stays on a white card. */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            background: var(--varo-panel);
            border: 1px solid var(--varo-line);
            border-radius: var(--varo-radius-card);
        }}
        /* The grid draws its own 8px corner inside our wrapper, which showed as a
           third radius wherever a table sits on the page. */
        [data-testid="stDataFrameResizable"],
        [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
        [data-testid="stDataFrame"] .stDataFrameGlideDataEditor,
        [data-testid="stDataFrame"] > div {{
            border-radius: var(--varo-radius-card) !important;
        }}
        /* Radio / checkbox labels readable */
        [data-testid="stWidgetLabel"], .stRadio label, .stCheckbox label {{ color: var(--varo-text) !important; }}
        [data-testid="stWidgetLabel"] p {{
            font-size: var(--varo-fs-secondary);
            font-weight: 600;
        }}
        /* Sidebar navigation */
        .v2-sidenav-title {{
            font-weight: var(--varo-fw-card);
            font-size: var(--varo-fs-card);
            color: var(--varo-text);
            margin: 0.1rem 0 0.5rem;
        }}
        /* The current screen is not signalled by colour alone: the active entry
           also carries a left rule and a heavier label. */
        [data-testid="stSidebar"] .stButton button[kind="primary"] {{
            border-left: 3px solid var(--varo-accent) !important;
            font-weight: 800 !important;
        }}
        [data-testid="stSidebar"] .stButton button {{
            justify-content: flex-start;
            text-align: left;
            font-size: var(--varo-fs-body);
        }}
        /* Home result dashboard helpers */
        .v2-home-badges {{ display: flex; flex-wrap: wrap; gap: 0.34rem; justify-content: flex-end; }}
        .v2-home-badge {{
            font-size: var(--varo-fs-caption);
            color: var(--varo-muted);
            background: #f1f4f7;
            border: 1px solid var(--varo-line);
            border-radius: 999px;
            padding: 0.12rem 0.55rem;
            white-space: nowrap;
        }}
        .v2-home-value {{ font-size: var(--varo-fs-metric); font-weight: var(--varo-fw-metric); line-height: 1.05; }}
        /* The four-step progress strip is gone with the second dashboard it sat
           on: the one flow now lives on the 재고 운영 screen itself. */
        .stTabs [data-baseweb="tab-list"] {{ flex-wrap: wrap; gap: 0.35rem; overflow-x: visible; }}
        .stTabs [data-baseweb="tab"] {{
            min-width: max-content;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
            white-space: normal;
        }}
        /* ------------------------------------------------------------------ */
        /* The one operations screen: status, network, decision                  */
        /* Heights are content-driven everywhere so Korean text never clips.    */
        /* ------------------------------------------------------------------ */
        /* A slim title strip: 데이터 상태 and 분석 상태 are already in the top bar,
           so repeating them here only pushed the network below the fold. */
        .ws-header {{ margin: 0.3rem 0 0.6rem; }}
        /* The workspace heading is a page title like the other three screens, so it
           uses the page level rather than a size of its own. */
        .ws-header-title {{
            font-size: var(--varo-fs-page);
            font-weight: var(--varo-fw-page);
            line-height: 1.25;
            letter-spacing: -0.01em;
            color: var(--varo-text);
        }}
        /* The 데이터 교체 toggle is a secondary control; it keeps its own row but
           not a full block's worth of vertical space above the result. */
        .st-key-quick_replace_bar {{ margin-top: -0.35rem; margin-bottom: -0.5rem; }}
        /* Tertiary: a quiet link-weight control tucked under the top bar rather
           than a third button competing with the page's real actions. */
        .st-key-quick_replace_bar button {{
            min-height: 30px !important;
            padding: 0.1rem 0.5rem !important;
            border-color: transparent !important;
            background: transparent !important;
            color: var(--varo-muted) !important;
        }}
        .st-key-quick_replace_bar button p {{
            font-size: var(--varo-fs-caption) !important;
            font-weight: 600;
        }}
        .st-key-quick_replace_bar button:hover {{
            background: var(--varo-panel-soft) !important;
            color: var(--varo-text) !important;
        }}
        /* Korean captions must not break inside a word ("...이동이 / 력"). */
        [data-testid="stCaptionContainer"] p {{
            word-break: keep-all;
            overflow-wrap: anywhere;
            font-size: var(--varo-fs-caption);
            line-height: 1.5;
        }}
        /* The three column headings of the workspace row are the same level as a
           section title anywhere else in the app. */
        .ws-panel-title {{
            font-size: var(--varo-fs-section);
            font-weight: var(--varo-fw-section);
            color: var(--varo-text);
            margin: var(--varo-section-gap) 0 var(--varo-block-gap);
        }}
        /* Headings *inside* a panel sit one level down, with the card titles. */
        .ws-block-title {{
            font-size: var(--varo-fs-card);
            font-weight: var(--varo-fw-card);
            color: var(--varo-text);
            margin: 0.85rem 0 0.32rem;
        }}
        .ws-side-card {{ padding: 0.8rem 0.85rem; }}
        .ws-side-value {{
            font-size: var(--varo-fs-lead);
            font-weight: var(--varo-fw-value);
            line-height: 1.35;
            color: var(--varo-text);
            margin-bottom: 0.22rem;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .ws-side-line {{
            margin-top: 0.3rem;
            font-size: var(--varo-fs-secondary);
            font-weight: 600;
            line-height: 1.45;
            color: var(--varo-text);
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .ws-count {{ color: var(--varo-muted); font-weight: 600; font-size: var(--varo-fs-secondary); }}
        /* The move list is one bounded scrolling group, so its height does not grow
           with the plan. The current row is not signalled by colour alone: it keeps
           a real checked radio, a heavier name and a left rule. Unselected rows stay
           plain lines rather than cards, so a long list never reads as heavy.
           (Comments here ship inside the <style> block, so they stay in English —
           product wording written here surfaces in AppTest's markdown.) */
        .st-key-ws_plan_pick [role="radiogroup"] {{ gap: 0.1rem; }}
        .st-key-ws_plan_pick [role="radiogroup"] > label {{
            align-items: flex-start;
            padding: 0.34rem 0.4rem 0.34rem 0.3rem;
            border-left: 3px solid transparent;
            border-radius: var(--varo-radius-button);
        }}
        .st-key-ws_plan_pick [role="radiogroup"] > label:hover {{ background: #f3f6f9; }}
        .st-key-ws_plan_pick [role="radiogroup"] > label:has(input:checked) {{
            background: #eef4fa;
            border-left-color: {DESIGN_TOKENS['accent']};
        }}
        .st-key-ws_plan_pick [role="radiogroup"] > label p {{
            font-size: var(--varo-fs-secondary);
            font-weight: 600;
            line-height: 1.4;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .st-key-ws_plan_pick [role="radiogroup"] > label:has(input:checked) p {{ font-weight: 800; }}
        .st-key-ws_plan_pick [data-testid="stCaptionContainer"] p {{
            font-size: var(--varo-fs-caption);
            line-height: 1.4;
            font-weight: 500;
        }}
        .ws-kpi {{ padding: var(--varo-card-padding); min-height: 0; }}
        .ws-kpi-title {{
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
            font-weight: 700;
        }}
        .ws-kpi-value {{
            font-size: var(--varo-fs-metric);
            font-weight: var(--varo-fw-metric);
            line-height: 1.2;
            color: var(--varo-text);
            margin-top: 0.2rem;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .ws-kpi-caption {{
            color: var(--varo-muted);
            font-size: var(--varo-fs-caption);
            line-height: 1.4;
            margin-top: 0.34rem;
        }}
        .ws-check-value {{
            font-size: var(--varo-fs-lead);
            font-weight: var(--varo-fw-value);
            line-height: 1.35;
            color: var(--varo-text);
            margin-top: 0.2rem;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .ws-network-shell {{
            width: 100%;
            border: 1px solid var(--varo-line);
            border-radius: var(--varo-radius-card);
            background: #fbfcfd;
            overflow: hidden;
        }}
        /* The SVG scales to the centre column, so every size below is multiplied
           by (column width / 940) on screen. Measured in a browser at 1366×768:
           the centre column is ~630px, a 0.67x downscale, so 19.5 lands at ~13px
           and 15 (the node-name floor in workspace_network) at ~10px.
           The ratio here is only the default for the smallest picture; a denser
           plan needs a taller canvas and writes its own aspect-ratio inline, so
           growing the canvas costs vertical room and never one glyph of size. */
        .ws-network-svg {{
            display: block;
            width: 100%;
            height: auto;
            aspect-ratio: 940 / 620;
        }}
        .ws-network-svg text {{ font-family: inherit; fill: var(--varo-text); }}
        /* Size only: the component measures each name against its own box and
           writes the result inline, which has to win over this default — a
           stylesheet rule outranks an SVG font-size *attribute*, which is how
           long store names used to be painted past the edge of their box. */
        .ws-network-svg .ws-node-name {{ font-size: 19.5px; font-weight: 760; }}
        .ws-network-svg .ws-node-sub {{ font-size: 15px; fill: var(--varo-muted); }}
        .ws-network-svg .ws-node-role {{ font-size: 16px; font-weight: 800; fill: {DESIGN_TOKENS['accent']}; }}
        .ws-network-svg .ws-edge-label {{ font-size: 17px; font-weight: 800; }}
        .ws-network-svg .ws-node {{ filter: drop-shadow(0 1px 2px rgba(30, 41, 59, 0.06)); }}
        /* Emphasis, not visibility: a node outside the current move stays fully
           drawn and keeps its name, it simply stops competing for attention. */
        .ws-network-svg .ws-node-context {{ opacity: 0.72; filter: none; }}
        .ws-network-svg .ws-node-context .ws-node-name {{ fill: var(--varo-muted); font-weight: 620; }}
        .ws-network-svg .ws-node-focus {{ filter: drop-shadow(0 2px 5px rgba(29, 111, 163, 0.22)); }}
        .ws-network-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
            padding: 0.55rem 0.75rem 0.15rem;
            color: var(--varo-muted);
            font-size: var(--varo-fs-caption);
        }}
        .ws-legend-item {{ display: inline-flex; align-items: center; gap: 0.26rem; }}
        .ws-legend-line {{ width: 26px; height: 0; border-top: 2px solid #8fa3b5; display: inline-block; }}
        .ws-legend-line-selected {{ border-top: 3px solid #1d6fa3; }}
        .ws-legend-line-dashed {{ border-top-style: dashed; }}
        .ws-legend-dot {{ width: 9px; height: 9px; border-radius: 50%; border: 1px solid; display: inline-block; }}
        .ws-legend-shape {{ width: 12px; height: 12px; display: inline-block; overflow: visible; }}
        .ws-network-placeholder {{
            min-height: 220px;
            display: grid;
            place-items: center;
            text-align: center;
            padding: 1.4rem 1rem;
            color: var(--varo-muted);
            border: 1px dashed var(--varo-line);
            border-radius: var(--varo-radius-card);
            background: #fbfcfd;
            margin-top: 0.5rem;
        }}
        /* The empty workspace reuses the placeholder box, but fills it with the
           order of the work instead of a second sentence saying it is empty. It
           borrows the existing type scale and the accent token — no new colour,
           no new radius — and the numbers are the badge pill at button size. */
        .ws-start-guide {{ text-align: left; align-content: center; justify-items: center; }}
        .ws-start-lead {{
            font-size: var(--varo-fs-body);
            color: var(--varo-muted);
            text-align: center;
            max-width: 46rem;
            margin: 0 auto 0.9rem;
            line-height: 1.5;
            word-break: keep-all;
        }}
        .ws-start-steps {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.7rem 1.5rem;
            /* Kept together rather than stretched across a 1500px card: four
               steps spread edge to edge stop reading as one sequence. 64rem is
               the width at which all four still share one line inside the
               narrowest supported card (1366 with the sidebar open); below that
               they wrap two and two, which still reads in order. */
            max-width: 64rem;
            margin: 0 auto;
            padding: 0;
        }}
        /* Streamlit indents every markdown <li> (an 18px left margin plus a
           padding) with an element-qualified rule, so a bare class loses to it.
           The selector is qualified with the element too — using this file's own
           class names, never a Streamlit-generated one — because this row is a
           step strip, not a bulleted list, and four steps only share one line
           once the four inherited indents are gone. */
        .ws-start-steps li.ws-start-step {{
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            max-width: 15rem;
            margin: 0;
            padding: 0;
        }}
        .ws-start-step-badge {{
            flex: 0 0 auto;
            width: 1.35rem;
            height: 1.35rem;
            border-radius: 999px;
            background: var(--varo-accent-soft);
            border: 1px solid var(--varo-accent-border);
            color: var(--varo-accent);
            font-size: var(--varo-fs-caption);
            font-weight: 700;
            display: grid;
            place-items: center;
            line-height: 1;
        }}
        .ws-start-step-body {{ display: block; }}
        .ws-start-step-body strong {{
            display: block;
            font-size: var(--varo-fs-secondary);
            font-weight: var(--varo-fw-value);
            color: var(--varo-text);
        }}
        .ws-start-step-body em {{
            display: block;
            font-style: normal;
            font-size: var(--varo-fs-caption);
            color: var(--varo-muted);
            line-height: 1.45;
            margin-top: 0.1rem;
            word-break: keep-all;
        }}
        /* The one card the whole screen exists to deliver. Its internal order is
           reading order: route, product, quantity, then the supporting values. */
        .ws-action-card {{ padding: 1rem 1.05rem; }}
        .ws-action-route {{
            font-size: var(--varo-fs-lead);
            font-weight: var(--varo-fw-lead);
            line-height: 1.35;
            color: var(--varo-text);
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .ws-action-product {{
            color: var(--varo-muted);
            font-size: var(--varo-fs-secondary);
            margin-top: 0.2rem;
        }}
        .ws-action-qty {{
            font-size: 2.05rem;
            font-weight: var(--varo-fw-metric);
            line-height: 1.15;
            color: {DESIGN_TOKENS['accent']};
            margin: 0.42rem 0 0.15rem;
        }}
        .ws-action-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem 0.7rem;
            margin-top: 0.65rem;
            padding-top: 0.65rem;
            border-top: 1px solid var(--varo-line);
        }}
        .ws-action-grid span {{
            display: block;
            color: var(--varo-muted);
            font-size: var(--varo-fs-caption);
        }}
        .ws-action-grid strong {{
            display: block;
            margin-top: 0.16rem;
            font-size: var(--varo-fs-body);
            font-weight: 700;
            color: var(--varo-text);
            line-height: 1.35;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        /* The money answer under the quantity gets the one step of emphasis
           between planned_qty and the remaining two values: all four were the
           same size before and nothing led the eye. */
        .ws-action-grid .ws-action-effect strong {{
            font-size: var(--varo-fs-lead);
            font-weight: var(--varo-fw-lead);
        }}
        /* The reason and risk bullets came out of Streamlit's markdown at 16px/400,
           larger than both their own heading and the decision values above them.
           They are supporting text and now read like it. */
        .st-key-{main_row} [data-testid="stColumn"]:nth-child(3) [data-testid="stMarkdownContainer"] ul {{
            margin: 0.1rem 0 0;
            padding-left: 1.05rem;
        }}
        .st-key-{main_row} [data-testid="stColumn"]:nth-child(3) [data-testid="stMarkdownContainer"] li {{
            font-size: var(--varo-fs-secondary);
            line-height: 1.5;
            color: var(--varo-text);
            margin-bottom: 0.22rem;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .st-key-{main_row} [data-testid="stColumn"]:nth-child(3) [data-testid="stMarkdownContainer"] li::marker {{
            color: var(--varo-muted);
        }}
        @media (max-width: 1400px) {{
            .ws-action-qty {{ font-size: 1.8rem; }}
            .ws-kpi-value {{ font-size: 1.5rem; }}
            .ws-action-grid .ws-action-effect strong {{ font-size: 1rem; }}
        }}
        /* ---- narrow desktops, and any desktop with the sidebar open ----------
           On a 1366-wide screen the expanded sidebar took 300px out of the window,
           which left the centre network ~547px wide. Because the SVG scales to its
           column (column width / 940), that pushed its smallest labels to 8.7-9.9px
           on screen -- measured in Chrome, not estimated.

           Three changes, none of which shrinks anything on screen:
             . the expanded sidebar is narrowed to 196px (it holds four short
               labels); the *collapsed* sidebar is untouched, hence [aria-expanded],
             . the page gutters shrink from 2rem to 0.9rem,
             . the three columns of the main row re-balance toward the centre
               (20 / 60 / 20) instead of 22 / 52 / 26.

           The ceiling is 1680 rather than 1450 because 1450 was the wrong bound:
           measured in Chrome at 1600x900 with the sidebar open, the centre column
           came out at 569px -- *narrower* than 1366 gets with this block applied
           (646px) -- and every SVG label landed at 9.2-9.7px. 1680 is the width
           above which an open 300px sidebar still leaves the centre column wide
           enough on its own, so the two sides of the bound meet without a cliff.

           The :has() guard restricts the ratio change to the one three-column row;
           the two-column rows nested inside it keep their own widths. */
        @media (max-width: 1680px) {{
            section[data-testid="stSidebar"][aria-expanded="true"] {{
                width: 196px !important;
                min-width: 196px !important;
                max-width: 196px !important;
            }}
            section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarContent"] {{
                padding-left: 0.55rem;
                padding-right: 0.55rem;
            }}
            .block-container {{ padding-left: 0.9rem !important; padding-right: 0.9rem !important; }}
            .st-key-{main_row} [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))
                > [data-testid="stColumn"] {{ min-width: 0 !important; }}
            /* Grow factors, not percentages: the browser shares out what is left
               *after* the two column gaps, so the third (decision) column can never
               be pushed past the right edge and the row never wraps. */
            .st-key-{main_row} [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))
                > [data-testid="stColumn"]:nth-child(1) {{ flex: 20 1 0% !important; width: auto !important; }}
            .st-key-{main_row} [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))
                > [data-testid="stColumn"]:nth-child(2) {{ flex: 60 1 0% !important; width: auto !important; }}
            .st-key-{main_row} [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(3))
                > [data-testid="stColumn"]:nth-child(3) {{ flex: 20 1 0% !important; width: auto !important; }}
            /* Only on narrow desktops: the two smallest SVG labels grow a little.
               One sits outside its box and the other is a single centred word, so
               neither can collide with anything. */
            .ws-network-svg .ws-node-sub {{ font-size: 16.5px; }}
            .ws-network-svg .ws-node-role {{ font-size: 17.5px; }}
        }}
        @media (max-width: 1150px) {{
            .ws-action-grid {{ grid-template-columns: 1fr; }}
            .ws-kpi-value {{ font-size: 1.34rem; }}
        }}
        @media (max-width: 1100px) {{
            .v2-kpi-card:not(.v2-kpi-card-compact) {{ min-height: 104px; }}
            .v2-kpi-value {{ font-size: 1.28rem; }}
            .v2-kpi-card-compact .v2-kpi-value {{ font-size: 1.5rem; }}
        }}
        @media (max-width: 920px) {{
            .v2-topbar {{ grid-template-columns: 1fr; align-items: start; }}
            .v2-topbar-meta {{ justify-content: flex-start; }}
            .v2-page-header {{ flex-direction: column; }}
            .v2-card-head {{ align-items: flex-start; flex-direction: column; }}
            .v2-recommendation-info {{ grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 0.7rem; }}
            .v2-info-item {{ border-right: 0; padding: 0 0.4rem; }}
            .v2-network-shell {{ min-height: 470px; }}
            .v2-network-svg {{ height: 470px; }}
        }}
        @media (max-width: 640px) {{
            .block-container {{ padding-left: 0.85rem; padding-right: 0.85rem; }}
            /* The page title keeps its one size at every width: 24px already fits
               a phone, and a second value here is how the four screens drifted
               apart in the first place. */
            .v2-kpi-value {{ font-size: 1.22rem; }}
            .v2-kpi-card-compact .v2-kpi-value {{ font-size: 1.24rem; }}
            .v2-card {{ padding: 0.82rem; }}
            .v2-kpi-card {{ min-height: 96px; padding: 0.62rem 0.68rem; }}
            .v2-detail-row {{ grid-template-columns: 1fr; gap: 0.2rem; }}
            .v2-recommendation-info {{ grid-template-columns: 1fr; }}
            .v2-network-shell {{ min-height: 360px; }}
            .v2-network-svg {{ height: 360px; }}
            .v2-network-svg .node-label {{ font-size: 11.4px; }}
            .v2-network-svg .dc-label {{ font-size: 12.6px; }}
            .stTabs [data-baseweb="tab-list"] {{
                display: flex;
                flex-wrap: nowrap;
                overflow-x: auto;
                scrollbar-width: thin;
                padding-bottom: 0.2rem;
            }}
            .stTabs [data-baseweb="tab"] {{
                min-width: max-content;
                padding-left: 0.48rem;
                padding-right: 0.48rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
