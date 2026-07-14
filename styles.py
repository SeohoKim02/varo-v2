"""Shared visual system for Varo V2."""
from __future__ import annotations

import streamlit as st

DESIGN_TOKENS = {
    "app_bg": "#f6f7f9",
    "card_bg": "#ffffff",
    "text": "#1b2533",
    "muted_text": "#4b5563",
    "border": "#c7d0dc",
    "accent": "#1f766d",
    "accent_soft": "#e8f4f2",
    "success": "#1f8a5b",
    "warning": "#c99700",
    "error": "#c2412d",
    "info": "#2d5f9a",
    "card_radius": "8px",
    "button_radius": "7px",
    "shadow": "0 8px 22px rgba(21, 30, 42, 0.08)",
}


def apply_global_styles() -> None:
    """Apply scoped V2 styles."""
    st.markdown(
        f"""
        <style>
        :root {{
            --varo-bg: {DESIGN_TOKENS['app_bg']};
            --varo-panel: {DESIGN_TOKENS['card_bg']};
            --varo-text: {DESIGN_TOKENS['text']};
            --varo-muted: {DESIGN_TOKENS['muted_text']};
            --varo-line: {DESIGN_TOKENS['border']};
            --varo-accent: {DESIGN_TOKENS['accent']};
            --varo-accent-soft: {DESIGN_TOKENS['accent_soft']};
            --varo-success: {DESIGN_TOKENS['success']};
            --varo-warning: {DESIGN_TOKENS['warning']};
            --varo-error: {DESIGN_TOKENS['error']};
            --varo-info: {DESIGN_TOKENS['info']};
            --varo-radius-card: {DESIGN_TOKENS['card_radius']};
            --varo-radius-button: {DESIGN_TOKENS['button_radius']};
            --varo-shadow: {DESIGN_TOKENS['shadow']};
        }}
        .stApp {{
            background: var(--varo-bg);
            color: var(--varo-text);
        }}
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stBottomBlockContainer"] {{
            background: #ffffff !important;
            color: var(--varo-text) !important;
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stNumberInput"] input,
        textarea,
        details,
        [data-testid="stDataFrame"] {{
            background: #ffffff !important;
            color: var(--varo-text) !important;
            color-scheme: light !important;
            border: 1px solid var(--varo-line);
            border-radius: var(--varo-radius-card);
            box-shadow: 0 4px 14px rgba(21, 30, 42, 0.04);
        }}
        details > summary,
        [data-testid="stExpander"] summary {{
            background: #ffffff !important;
            color: var(--varo-text) !important;
            font-weight: 720 !important;
            border-color: var(--varo-line) !important;
        }}
        [data-testid="stDataFrame"] canvas,
        [data-testid="stDataFrame"] [role="grid"] {{
            color-scheme: light !important;
            background: #ffffff !important;
            color: var(--varo-text) !important;
        }}
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [role="listbox"],
        [role="menu"],
        [role="option"] {{
            background: #ffffff !important;
            color: var(--varo-text) !important;
        }}
        html body .stApp label[data-baseweb="checkbox"] > span:first-child {{
            background: #ffffff !important;
            background-color: #ffffff !important;
            border: 1px solid #94a3b8 !important;
        }}
        html body .stApp label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {{
            background: #dbeafe !important;
            background-color: #dbeafe !important;
            border-color: #7db2ea !important;
        }}
        html body .stApp [data-testid="stElementToolbar"],
        html body .stApp [data-testid="stElementToolbarButtonContainer"],
        html body .stApp [data-testid="stElementToolbarButton"] {{
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: var(--varo-text) !important;
        }}
        .stButton button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="baseButton-secondary"] {{
            background: #ffffff !important;
            color: var(--varo-text) !important;
            border-color: #aeb9c7 !important;
            font-weight: 680 !important;
        }}
        .block-container {{
            padding-top: 2.55rem !important;
            padding-bottom: 3rem;
            max-width: 1520px;
            margin-left: auto !important;
            margin-right: auto !important;
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
            font-size: 1.08rem;
            font-weight: 780;
            letter-spacing: 0;
            color: var(--varo-text);
        }}
        .v2-page-context {{
            margin-top: 0.12rem;
            color: #3f4c5d;
            font-size: 0.78rem;
            font-weight: 680;
        }}
        .v2-topbar-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            justify-content: flex-end;
            align-items: center;
            color: var(--varo-muted);
            font-size: 0.86rem;
        }}
        .v2-pill, .v2-file-label {{
            border: 1px solid var(--varo-line);
            border-radius: 999px;
            padding: 0.22rem 0.56rem;
            background: #f8fafc;
        }}
        .v2-file-label {{
            max-width: min(46vw, 620px);
            white-space: normal;
            overflow-wrap: anywhere;
        }}
        .v2-pill {{
            color: var(--varo-accent);
            background: var(--varo-accent-soft);
            border-color: #cde6e1;
            font-weight: 700;
        }}
        .v2-data-onboarding {{
            border: 1px solid #cde6e1;
            background: #f7fbfa;
            border-radius: var(--varo-radius-card);
            padding: 0.62rem 0.78rem;
            margin: 0.25rem 0 0.4rem;
        }}
        .v2-data-title {{
            color: var(--varo-text);
            font-size: 0.94rem;
            font-weight: 740;
            margin-bottom: 0.12rem;
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
            font-size: 0.84rem;
            overflow-wrap: anywhere;
        }}
        .v2-page-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.8rem;
            margin: 0.48rem 0 0.62rem;
        }}
        .v2-page-title {{
            font-size: 1.42rem !important;
            font-weight: 760;
            line-height: 1.22 !important;
            color: var(--varo-text);
            margin: 0 !important;
            padding: 0 !important;
        }}
        .v2-page-desc {{
            margin-top: 0.2rem;
            color: var(--varo-muted);
            font-size: 0.9rem;
            line-height: 1.4;
        }}
        .v2-card {{
            border: 1px solid var(--varo-line);
            background: var(--varo-panel);
            border-radius: var(--varo-radius-card);
            box-shadow: var(--varo-shadow);
            padding: 0.9rem;
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
            padding: 0.95rem 1rem;
            min-height: 120px;
        }}
        .v2-kpi-card-compact {{
            min-height: 118px;
            padding: 0.92rem 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow: visible;
        }}
        .v2-kpi-card-compact .v2-kpi-value {{
            font-size: clamp(1.28rem, 1.65vw, 1.72rem);
            font-weight: 820;
            margin-top: 0.28rem;
        }}
        .v2-kpi-value-file {{
            font-size: clamp(1rem, 1.25vw, 1.24rem) !important;
            line-height: 1.28 !important;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            overflow-wrap: anywhere;
        }}
        .v2-card-title {{
            color: var(--varo-text);
            font-weight: 720;
            font-size: 1rem;
            margin-bottom: 0.42rem;
        }}
        .v2-card-caption {{
            color: var(--varo-muted);
            font-size: 0.86rem;
            line-height: 1.4;
        }}
        .v2-kpi-value {{
            font-size: 1.42rem;
            font-weight: 800;
            line-height: 1.12;
            color: var(--varo-text);
            margin-top: 0.12rem;
        }}
        .v2-section-header {{
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            margin: 0.9rem 0 0.42rem;
        }}
        .v2-section-title {{
            font-size: 1.04rem;
            font-weight: 740;
            color: var(--varo-text);
        }}
        .v2-section-desc {{
            color: var(--varo-muted);
            font-size: 0.82rem;
            margin-top: 0.08rem;
        }}
        .v2-empty-state, .v2-error-card {{
            border: 1px dashed var(--varo-line);
            background: #fbfcfd;
            border-radius: var(--varo-radius-card);
            padding: 0.92rem 1rem;
            color: var(--varo-muted);
            min-height: 68px;
        }}
        .v2-empty-state-compact {{ min-height: 0; padding: 0.72rem 0.82rem; }}
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
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1.15;
            border: 1px solid var(--varo-line);
            white-space: normal;
        }}
        .v2-badge-neutral {{ background: #f3f5f7; color: var(--varo-muted); }}
        .v2-badge-accent {{ background: var(--varo-accent-soft); color: #155f57; border-color: #9fcfc7; }}
        .v2-badge-success {{ background: #e8f7ef; color: #176b47; border-color: #9fd5b8; }}
        .v2-badge-warning {{ background: #fff6d7; color: #6f5000; border-color: #d8bb52; }}
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
        .v2-info-item strong {{
            display: block;
            margin-top: 0.2rem;
            color: var(--varo-text);
            font-size: 0.95rem;
            line-height: 1.4;
            white-space: normal;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }}
        .v2-table-shell {{
            border: 1.25px solid #b9c4d1;
            background: var(--varo-panel);
            border-radius: var(--varo-radius-card);
            overflow: visible;
        }}
        [data-testid="stDataFrame"] {{
            border: 1.25px solid #b9c4d1 !important;
        }}
        table thead tr, table thead th {{
            background: #eef2f6 !important;
            color: #1f2937 !important;
            font-weight: 760 !important;
            border-color: #b9c4d1 !important;
        }}
        .v2-quick-nav-card {{
            min-height: 88px;
            border: 1.25px solid #b8c7d8;
            background: #ffffff;
            border-radius: var(--varo-radius-card);
            box-shadow: 0 5px 15px rgba(30, 58, 95, 0.07);
            padding: 0.82rem 0.9rem;
            margin-top: 0.12rem;
        }}
        .v2-network-shell {{
            position: relative;
            width: 100%;
            min-height: 760px;
            border: 1.4px solid #b9c4d1;
            border-radius: var(--varo-radius-card);
            background: radial-gradient(circle at 52% 46%, #ffffff 0, #f8fafb 58%, #f3f6f8 100%);
            overflow: hidden;
        }}
        .v2-network-placeholder {{
            min-height: 660px;
            display: grid;
            place-items: center;
            color: var(--varo-muted);
            text-align: center;
            padding: 1rem;
        }}
        .v2-network-svg {{ display: block; width: 100%; height: 700px; margin-top: 0.2rem; }}
        .v2-network-svg text {{ font-family: inherit; fill: var(--varo-text); }}
        .v2-network-svg .node-label {{
            font-size: 14px;
            font-weight: 780;
            paint-order: stroke;
            stroke: rgba(255, 255, 255, 0.92);
            stroke-width: 2.4px;
            stroke-linejoin: round;
        }}
        .v2-network-svg .dc-label {{ font-size: 16.4px; font-weight: 820; }}
        .v2-network-svg .node-type {{ font-size: 10.5px; fill: var(--varo-muted); }}
        .v2-network-svg .network-node {{ filter: drop-shadow(0 2px 3px rgba(30, 41, 59, 0.06)); }}
        .v2-network-svg .recommended-node {{ filter: drop-shadow(0 3px 5px rgba(216, 131, 120, 0.18)); }}
        .v2-network-svg .v2-vehicle {{ filter: drop-shadow(0 3px 5px rgba(30, 41, 59, 0.20)); }}
        .v2-network-svg .v2-vehicle-selected {{ opacity: 1; }}
        .v2-network-svg .v2-vehicle-muted {{ opacity: 0.56; }}
        .v2-network-svg .vehicle-route {{ font-size: 11px; font-weight: 840; }}
        .v2-network-svg .vehicle-mode {{ font-size: 9.4px; font-weight: 780; }}
        .v2-network-svg .vehicle-type {{ font-size: 8px; font-weight: 800; fill: #ffffff; }}
        .v2-network-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem 0.9rem;
            align-items: center;
            padding: 0.72rem 0.85rem 0.2rem;
            color: #344154;
            font-size: 0.84rem;
            font-weight: 680;
        }}
        .v2-legend-item {{ display: inline-flex; align-items: center; gap: 0.36rem; }}
        .v2-legend-line {{ width: 32px; height: 0; border-top: 3px solid #2563a6; }}
        .v2-legend-line-via {{ border-top-color: #c28a00; border-top-style: dashed; }}
        .v2-legend-node {{ width: 21px; height: 14px; border-radius: 3px; display: inline-block; }}
        .v2-legend-store {{ background: #eef5ff; border: 2px solid #78a9d8; }}
        .v2-legend-dc {{ background: #fff8df; border: 2px solid #c28a00; }}
        .v2-legend-truck {{ font-size: 1.08rem; line-height: 1; }}
        .v2-route-steps {{
            margin: 0;
            padding-left: 1.25rem;
            color: var(--varo-text);
            line-height: 1.75;
        }}
        .v2-route-flow {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.7rem;
            flex-wrap: wrap;
            border: 1.25px solid #b9c4d1;
            background: #f8fafc;
            border-radius: var(--varo-radius-card);
            padding: 1rem;
            margin-bottom: 0.55rem;
        }}
        .v2-route-node {{
            max-width: 280px;
            padding: 0.58rem 0.8rem;
            border: 2px solid #78a9d8;
            background: #eef5ff;
            border-radius: 7px;
            color: #24364e;
            font-weight: 740;
            text-align: center;
        }}
        .v2-route-node-dc {{ border-color: #c28a00; background: #fff8df; color: #624900; }}
        .v2-route-arrow {{ color: #526274; font-size: 1.35rem; font-weight: 800; }}
        .v2-route-code {{ color: var(--varo-muted); font-size: 0.7rem; margin-left: 0.25rem; }}
        .stButton button[kind="primary"],
        button[data-testid="stBaseButton-primary"],
        button[data-testid="baseButton-primary"] {{
            background-color: #dbeafe !important;
            border-color: #93c5fd !important;
            color: #1e3a5f !important;
            box-shadow: none !important;
        }}
        .stButton button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {{
            background-color: #bfdbfe !important;
            border-color: #7db2ea !important;
        }}
        /* Sidebar navigation */
        .v2-sidenav-title {{
            font-weight: 760;
            font-size: 0.95rem;
            color: var(--varo-text);
            margin: 0.1rem 0 0.5rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{ flex-wrap: wrap; gap: 0.35rem; overflow-x: visible; }}
        .stTabs [data-baseweb="tab"] {{
            min-width: max-content;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
            white-space: normal;
            color: #344154 !important;
            font-weight: 700 !important;
        }}
        @media (max-width: 1100px) {{
            .v2-kpi-card:not(.v2-kpi-card-compact) {{ min-height: 104px; }}
            .v2-kpi-value {{ font-size: 1.28rem; }}
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
            .v2-page-title {{ font-size: 1.28rem; }}
            .v2-kpi-value {{ font-size: 1.22rem; }}
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
