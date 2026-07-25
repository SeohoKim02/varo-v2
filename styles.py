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
    "card_radius": "8px",
    "button_radius": "7px",
    "shadow": "0 6px 18px rgba(17, 24, 39, 0.05)",
}


def apply_global_styles() -> None:
    """Apply scoped V2 styles."""
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
            padding-top: 2.9rem !important;
            padding-bottom: 3rem;
            max-width: 1480px;
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
            font-size: 1.42rem;
            font-weight: 760;
            line-height: 1.22;
            color: var(--varo-text);
            margin: 0;
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
            font-size: 1.9rem;
            font-weight: 820;
            margin-top: 0.15rem;
        }}
        .v2-kpi-title {{
            color: var(--varo-muted);
            font-size: 0.9rem;
            font-weight: 700;
        }}
        .v2-kpi-desc {{
            color: var(--varo-muted);
            font-size: 0.8rem;
            line-height: 1.35;
            margin-top: 0.4rem;
        }}
        .v2-card-title {{
            color: var(--varo-text);
            font-weight: 720;
            font-size: 1rem;
            margin-bottom: 0.42rem;
        }}
        .v2-card-caption {{
            color: var(--varo-muted);
            font-size: 0.84rem;
            line-height: 1.4;
        }}
        .v2-kpi-value {{
            font-size: 1.42rem;
            font-weight: 800;
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
            margin: 0.68rem 0 0.36rem;
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
        /* Home status card: roomy, never fixed-height so Korean text/buttons never clip. */
        .v2-home-state-card {{
            padding: 1.15rem 1.25rem;
            min-height: 0;
        }}
        .v2-home-state-card .v2-card-title {{ font-size: 1.12rem; line-height: 1.4; }}
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
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1.15;
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
            font-size: 0.86rem;
            color: var(--varo-text);
        }}
        .v2-html-table thead th {{
            background: var(--varo-panel-soft);
            color: var(--varo-muted);
            font-weight: 700;
            text-align: left;
            padding: 0.5rem 0.62rem;
            border-bottom: 1px solid var(--varo-line);
            white-space: nowrap;
        }}
        .v2-html-table tbody td {{
            padding: 0.44rem 0.62rem;
            border-bottom: 1px solid var(--varo-line);
            color: var(--varo-text);
            white-space: nowrap;
        }}
        .v2-html-table tbody tr:nth-child(2n) td {{ background: #fcfdfe; }}
        .v2-html-table tbody tr:last-child td {{ border-bottom: 0; }}
        .v2-html-table tbody tr.v2-row-pick td {{ background: var(--varo-accent-soft); font-weight: 700; }}
        .v2-hbar {{ display: flex; flex-direction: column; gap: 0.5rem; margin: 0.3rem 0 0.2rem; }}
        .v2-hbar-row {{ display: grid; grid-template-columns: 120px 1fr 56px; gap: 0.6rem; align-items: center; }}
        .v2-hbar-label {{ color: var(--varo-text); font-size: 0.9rem; font-weight: 640; }}
        .v2-hbar-track {{ background: var(--varo-panel-soft); border: 1px solid var(--varo-line); border-radius: 999px; height: 16px; overflow: hidden; }}
        .v2-hbar-fill {{ display: block; height: 100%; background: var(--varo-accent-soft); border-right: 2px solid var(--varo-accent); }}
        .v2-hbar-value {{ color: var(--varo-muted); font-size: 0.9rem; text-align: right; }}
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
            font-size: 0.78rem;
        }}
        .v2-legend-line {{ width: 28px; height: 0; border-top: 2px solid var(--varo-accent); }}
        .v2-legend-line-dashed {{ border-top-style: dashed; }}
        .v2-legend-state {{ display: inline-flex; align-items: center; gap: 0.22rem; }}
        .v2-legend-dot {{ width: 9px; height: 9px; border-radius: 50%; border: 1px solid; display: inline-block; }}
        .v2-running-route {{
            border: 1px solid var(--varo-line);
            border-radius: 7px;
            padding: 0.5rem 0.55rem;
            margin-top: 0.42rem;
            background: #fbfcfd;
        }}
        .v2-running-route-selected {{ border: 2px solid var(--varo-accent); }}
        .v2-running-route strong {{ display: block; font-size: 0.88rem; line-height: 1.25; }}
        .v2-running-route-meta {{
            display: grid;
            gap: 0.32rem;
            margin-top: 0.28rem;
            color: var(--varo-muted);
            font-size: 0.76rem;
        }}
        .v2-route-code {{ color: var(--varo-muted); font-size: 0.7rem; margin-left: 0.25rem; }}
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
        /* Inputs / select / number */
        [data-baseweb="select"] > div, [data-baseweb="input"] {{
            background-color: var(--varo-panel) !important;
            border-color: var(--varo-line) !important;
        }}
        .stTextInput input, .stNumberInput input, [data-baseweb="input"] input,
        [data-baseweb="select"] input, textarea {{
            background-color: var(--varo-panel) !important;
            color: var(--varo-text) !important;
        }}
        [data-baseweb="popover"] div[role="listbox"], [data-baseweb="menu"] {{
            background: var(--varo-panel) !important;
        }}
        /* Tabs: selected tab gets a soft blue chip, dark text. */
        .stTabs [data-baseweb="tab"] {{ color: var(--varo-muted) !important; background: transparent; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: var(--varo-accent) !important;
            background: var(--varo-accent-soft) !important;
            border-radius: 7px 7px 0 0;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--varo-accent) !important; }}
        .stTabs [data-baseweb="tab-border"] {{ background-color: var(--varo-line) !important; }}
        /* Expander */
        [data-testid="stExpander"] {{
            background: var(--varo-panel);
            border: 1px solid var(--varo-line) !important;
            border-radius: var(--varo-radius-card);
        }}
        [data-testid="stExpander"] summary {{ background: var(--varo-panel-soft); color: var(--varo-text); }}
        [data-testid="stExpander"] summary:hover {{ color: var(--varo-accent); }}
        /* Metric — let long Korean status text wrap instead of being clipped. */
        [data-testid="stMetric"] {{ background: transparent; color: var(--varo-text); min-width: 0; }}
        [data-testid="stMetricValue"] {{
            color: var(--varo-strong);
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: keep-all;
            line-height: 1.22;
        }}
        [data-testid="stMetricValue"] > div {{ white-space: normal; overflow-wrap: anywhere; }}
        [data-testid="stMetricLabel"] {{ color: var(--varo-muted); }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] p {{ white-space: normal; overflow-wrap: anywhere; word-break: keep-all; }}
        /* Dataframe wrapper stays on a white card. */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            background: var(--varo-panel);
            border: 1px solid var(--varo-line);
            border-radius: 8px;
        }}
        /* Radio / checkbox labels readable */
        [data-testid="stWidgetLabel"], .stRadio label, .stCheckbox label {{ color: var(--varo-text) !important; }}
        /* Sidebar navigation */
        .v2-sidenav-title {{
            font-weight: 760;
            font-size: 0.95rem;
            color: var(--varo-text);
            margin: 0.1rem 0 0.5rem;
        }}
        /* Home result dashboard helpers */
        .v2-home-badges {{ display: flex; flex-wrap: wrap; gap: 0.34rem; justify-content: flex-end; }}
        .v2-home-badge {{
            font-size: 0.72rem;
            color: var(--varo-muted);
            background: #f1f4f7;
            border: 1px solid var(--varo-line);
            border-radius: 999px;
            padding: 0.12rem 0.55rem;
            white-space: nowrap;
        }}
        .v2-home-value {{ font-size: 1.9rem; font-weight: 820; line-height: 1.05; }}
        /* Thin single-row progress strip (엑셀 업로드 → 재고 분석 → 이동 추천 → 결과 확인). */
        .v2-flow-row {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin: 0.55rem 0 0.2rem;
            padding: 0.5rem 0.8rem;
            border: 1px solid var(--varo-line);
            background: var(--varo-panel-soft);
            border-radius: 999px;
            font-size: 0.9rem;
        }}
        .v2-flow-item {{ color: var(--varo-muted); font-weight: 600; }}
        .v2-flow-current {{
            color: var(--varo-accent); font-weight: 760;
            background: var(--varo-accent-soft); border: 1px solid var(--varo-accent-border);
            border-radius: 999px; padding: 0.1rem 0.62rem;
        }}
        .v2-flow-arrow {{ color: var(--varo-muted); font-weight: 700; }}
        @media (max-width: 640px) {{ .v2-flow-row {{ border-radius: 14px; }} }}
        .stTabs [data-baseweb="tab-list"] {{ flex-wrap: wrap; gap: 0.35rem; overflow-x: visible; }}
        .stTabs [data-baseweb="tab"] {{
            min-width: max-content;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
            white-space: normal;
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
            .v2-page-title {{ font-size: 1.28rem; }}
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
