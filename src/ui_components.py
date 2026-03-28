from __future__ import annotations

from html import escape
from types import SimpleNamespace

import pandas as pd
import plotly.express as px
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - test stubs may not expose graph_objects
    go = SimpleNamespace(Figure=object, layout=SimpleNamespace(Template=lambda: None), Layout=lambda **kwargs: kwargs)
try:
    import plotly.io as pio
except Exception:  # pragma: no cover - test stubs may not expose plotly.io
    pio = None
import streamlit as st


BRAND_TITLE = 'Clinverity'
BRAND_SUBTITLE = 'Clinical Data Quality, Remediation & Audit Platform'
BRAND_TAGLINE = 'Detect. Remediate. Validate. Explain.'

BRAND_COLORS = {
    'primary': '#1F4E79',
    'primary_dark': '#163a5a',
    'primary_soft': '#EAF2F8',
    'accent': '#14B8A6',
    'accent_soft': '#E7FBF8',
    'background': '#DCE8F2',
    'background_alt': '#CFE0ED',
    'card': '#E6F0F7',
    'text': '#0F172A',
    'muted': '#475569',
    'muted_soft': '#64748B',
    'border': '#E2E8F0',
    'success': '#16A34A',
    'success_soft': '#EAF8EF',
    'warning': '#F59E0B',
    'warning_soft': '#FFF5E6',
    'error': '#DC2626',
    'error_soft': '#FEECEC',
    'info': '#0EA5E9',
    'info_soft': '#E8F7FF',
    'shadow': '0 18px 42px rgba(15, 23, 42, 0.08)',
    'shadow_soft': '0 10px 26px rgba(15, 23, 42, 0.05)',
}

TYPOGRAPHY_SCALE = {
    'hero': '2.4rem',
    'h1': '2.05rem',
    'h2': '1.34rem',
    'h3': '1.08rem',
    'body': '0.96rem',
    'caption': '0.81rem',
}

SPACING_SCALE = {
    'xs': '0.35rem',
    'sm': '0.6rem',
    'md': '0.95rem',
    'lg': '1.25rem',
    'xl': '1.7rem',
}

SURFACE_RULES = {
    'radius_card': '18px',
    'radius_panel': '22px',
    'radius_input': '14px',
}

CARD_STYLE = f"""
<style>
    :root {{
        --clinverity-primary: {BRAND_COLORS['primary']};
        --clinverity-primary-dark: {BRAND_COLORS['primary_dark']};
        --clinverity-primary-soft: {BRAND_COLORS['primary_soft']};
        --clinverity-accent: {BRAND_COLORS['accent']};
        --clinverity-accent-soft: {BRAND_COLORS['accent_soft']};
        --clinverity-bg: {BRAND_COLORS['background']};
        --clinverity-bg-alt: {BRAND_COLORS['background_alt']};
        --clinverity-card: {BRAND_COLORS['card']};
        --clinverity-text: {BRAND_COLORS['text']};
        --clinverity-muted: {BRAND_COLORS['muted']};
        --clinverity-muted-soft: {BRAND_COLORS['muted_soft']};
        --clinverity-border: {BRAND_COLORS['border']};
        --clinverity-success: {BRAND_COLORS['success']};
        --clinverity-success-soft: {BRAND_COLORS['success_soft']};
        --clinverity-warning: {BRAND_COLORS['warning']};
        --clinverity-warning-soft: {BRAND_COLORS['warning_soft']};
        --clinverity-error: {BRAND_COLORS['error']};
        --clinverity-error-soft: {BRAND_COLORS['error_soft']};
        --clinverity-info: {BRAND_COLORS['info']};
        --clinverity-info-soft: {BRAND_COLORS['info_soft']};
        --clinverity-shadow: {BRAND_COLORS['shadow']};
        --clinverity-shadow-soft: {BRAND_COLORS['shadow_soft']};
        --clinverity-radius-card: {SURFACE_RULES['radius_card']};
        --clinverity-radius-panel: {SURFACE_RULES['radius_panel']};
        --clinverity-radius-input: {SURFACE_RULES['radius_input']};
        --clinverity-font-sans: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    .stApp {{
        background: var(--clinverity-bg);
        color: var(--clinverity-text);
        text-rendering: optimizeLegibility;
        font-family: var(--clinverity-font-sans);
        animation: clinverity-fade-in 0.24s ease-out;
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }}
    [data-testid='stHeader'] {{
        background: rgba(248, 250, 252, 0.92);
        border-bottom: 1px solid var(--clinverity-border);
    }}
    [data-testid='stSidebar'] {{
        background: #fdfefe;
        border-right: 1px solid var(--clinverity-border);
    }}
    [data-testid='stSidebar'] .block-container {{
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    h1, h2, h3, h4 {{
        color: var(--clinverity-text);
        letter-spacing: -0.02em;
        font-family: var(--clinverity-font-sans);
    }}
    h1 {{
        font-size: {TYPOGRAPHY_SCALE['h1']};
        font-weight: 700;
        margin-bottom: 0.35rem;
    }}
    h2 {{
        font-size: {TYPOGRAPHY_SCALE['h2']};
        margin-top: 1.2rem;
        margin-bottom: 0.45rem;
    }}
    h3 {{
        font-size: {TYPOGRAPHY_SCALE['h3']};
        margin-top: 1rem;
        margin-bottom: 0.35rem;
    }}
    p {{
        font-size: {TYPOGRAPHY_SCALE['body']};
        line-height: 1.58;
        margin-bottom: 0.55rem;
    }}
    p, li, div, label {{
        color: var(--clinverity-text);
    }}
    a {{
        color: var(--clinverity-primary);
    }}
    .stApp [data-testid='stVerticalBlock'] > [style*='flex-direction: column'] {{
        gap: 0.25rem;
    }}
    [data-testid='stMarkdownContainer'] ul,
    [data-testid='stMarkdownContainer'] ol {{
        padding-left: 1.2rem;
        margin-bottom: 0.75rem;
    }}
    [data-testid='stCaptionContainer'] p,
    .stCaption,
    small {{
        color: var(--clinverity-muted) !important;
    }}
    .clinverity-hero {{
        background:
            radial-gradient(circle at top right, rgba(20, 184, 166, 0.08), transparent 30%),
            linear-gradient(180deg, #f6fafe 0%, #edf5fb 100%);
        border: 1px solid var(--clinverity-border);
        border-radius: var(--clinverity-radius-panel);
        padding: 1.45rem 1.6rem 1.2rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--clinverity-shadow);
    }}
    .clinverity-hero-eyebrow {{
        color: var(--clinverity-primary);
        font-size: 0.79rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }}
    .clinverity-hero-title {{
        color: var(--clinverity-text);
        font-size: {TYPOGRAPHY_SCALE['hero']};
        font-weight: 750;
        line-height: 1.04;
        margin-bottom: 0.3rem;
    }}
    .clinverity-hero-subtitle {{
        color: var(--clinverity-primary);
        font-size: 1.02rem;
        font-weight: 650;
        margin-bottom: 0.25rem;
    }}
    .clinverity-hero-tagline {{
        color: var(--clinverity-muted);
        font-size: 0.92rem;
        margin-bottom: 0.95rem;
    }}
    .clinverity-context-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-top: 0.2rem;
    }}
    .clinverity-context-card {{
        background: rgba(248, 250, 252, 0.94);
        border: 1px solid var(--clinverity-border);
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }}
    .clinverity-context-card:hover {{
        transform: translateY(-1px);
        border-color: rgba(31, 78, 121, 0.22);
        box-shadow: var(--clinverity-shadow-soft);
    }}
    .clinverity-context-label {{
        color: var(--clinverity-muted);
        font-size: 0.76rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.28rem;
    }}
    .clinverity-context-value {{
        color: var(--clinverity-text);
        font-size: 0.98rem;
        font-weight: 650;
        line-height: 1.3;
    }}
    .clinverity-sidebar-brand {{
        background: linear-gradient(180deg, #f5f9fd 0%, #eaf2f9 100%);
        border: 1px solid var(--clinverity-border);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem;
        margin-bottom: 1rem;
        box-shadow: var(--clinverity-shadow-soft);
    }}
    .clinverity-sidebar-kicker {{
        color: var(--clinverity-primary);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }}
    .clinverity-sidebar-title {{
        color: var(--clinverity-text);
        font-size: 1.28rem;
        font-weight: 760;
        line-height: 1.05;
        margin-bottom: 0.22rem;
    }}
    .clinverity-sidebar-subtitle {{
        color: var(--clinverity-primary);
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 0.2rem;
    }}
    .clinverity-sidebar-tagline {{
        color: var(--clinverity-muted);
        font-size: 0.78rem;
        line-height: 1.35;
    }}
    .clinverity-sidebar-meta {{
        display: grid;
        gap: 0.45rem;
        margin-top: 0.9rem;
    }}
    .clinverity-sidebar-meta-item {{
        background: rgba(244,248,252,0.92);
        border: 1px solid var(--clinverity-border);
        border-radius: 12px;
        padding: 0.6rem 0.7rem;
    }}
    .clinverity-sidebar-meta-label {{
        color: var(--clinverity-muted);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 0.16rem;
    }}
    .clinverity-sidebar-meta-value {{
        color: var(--clinverity-text);
        font-size: 0.84rem;
        font-weight: 650;
        line-height: 1.35;
    }}
    .clinverity-sidebar-section {{
        color: var(--clinverity-muted);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0.95rem 0 0.45rem;
    }}
    .clinverity-sidebar-panel {{
        background: linear-gradient(180deg, #f5f9fd 0%, #ebf3f9 100%);
        border: 1px solid var(--clinverity-border);
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
        margin: 0.35rem 0 0.75rem;
        box-shadow: var(--clinverity-shadow-soft);
    }}
    .clinverity-sidebar-panel-title {{
        color: var(--clinverity-text);
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .clinverity-sidebar-panel-text {{
        color: var(--clinverity-muted);
        font-size: 0.8rem;
        line-height: 1.45;
    }}
    .clinverity-sidebar-stat-grid {{
        display: grid;
        gap: 0.45rem;
        margin-top: 0.55rem;
    }}
    .clinverity-sidebar-stat {{
        display: flex;
        flex-direction: column;
        gap: 0.08rem;
        padding: 0.55rem 0.65rem;
        border-radius: 12px;
        background: #f3f8fc;
        border: 1px solid var(--clinverity-border);
    }}
    .clinverity-sidebar-stat-label {{
        color: var(--clinverity-muted-soft);
        font-size: 0.68rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 700;
    }}
    .clinverity-sidebar-stat-value {{
        color: var(--clinverity-text);
        font-size: 0.84rem;
        font-weight: 700;
        line-height: 1.35;
    }}
    .clinverity-section-intro {{
        margin: 0.15rem 0 1rem;
        padding: 0.2rem 0 0.25rem;
    }}
    .clinverity-section-title {{
        color: var(--clinverity-text);
        font-size: 1.28rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .clinverity-section-description {{
        color: var(--clinverity-muted);
        font-size: 0.92rem;
        line-height: 1.5;
    }}
    .clinverity-subsection {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin: 1rem 0 0.55rem;
        padding-top: 0.15rem;
    }}
    .clinverity-subsection-copy {{
        min-width: 0;
    }}
    .clinverity-subsection-title {{
        color: var(--clinverity-text);
        font-size: 1.04rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }}
    .clinverity-subsection-note {{
        color: var(--clinverity-muted);
        font-size: 0.84rem;
        line-height: 1.45;
    }}
    .clinverity-badge-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.45rem 0 0.7rem;
    }}
    .clinverity-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.35rem 0.62rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid var(--clinverity-border);
        background: #f2f7fb;
        color: var(--clinverity-text);
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.04);
    }}
    .clinverity-badge[data-tone='info'] {{
        background: var(--clinverity-info-soft);
        color: #0369A1;
        border-color: rgba(14, 165, 233, 0.16);
    }}
    .clinverity-badge[data-tone='accent'] {{
        background: var(--clinverity-accent_soft);
        color: #0F766E;
        border-color: rgba(20, 184, 166, 0.16);
    }}
    .clinverity-badge[data-tone='success'] {{
        background: var(--clinverity-success-soft);
        color: #166534;
        border-color: rgba(22, 163, 74, 0.16);
    }}
    .clinverity-badge[data-tone='warning'] {{
        background: var(--clinverity-warning-soft);
        color: #9A6700;
        border-color: rgba(245, 158, 11, 0.16);
    }}
    .sda-metric-card {{
        background: var(--clinverity-card);
        border: 1px solid var(--clinverity-border);
        border-top: 4px solid var(--clinverity-primary);
        border-radius: var(--clinverity-radius-card);
        padding: 0.95rem 1rem 0.9rem;
        box-shadow: var(--clinverity-shadow-soft);
        min-height: 116px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.45rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }}
    .sda-metric-card:hover {{
        transform: translateY(-1px);
        box-shadow: var(--clinverity-shadow);
        border-color: rgba(31, 78, 121, 0.18);
    }}
    .sda-metric-label {{
        color: var(--clinverity-muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.35;
        white-space: normal;
        word-break: break-word;
    }}
    .sda-metric-value {{
        color: var(--clinverity-text);
        font-size: 1.45rem;
        font-weight: 760;
        line-height: 1.18;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
    }}
    .sda-metric-help {{
        color: var(--clinverity-muted);
        font-size: 0.79rem;
        line-height: 1.38;
        white-space: normal;
        word-break: break-word;
    }}
    div[data-testid='stMetricLabel'],
    div[data-testid='stMetricValue'] {{
        overflow: visible !important;
    }}
    div[data-testid='stMetricLabel'] > div,
    div[data-testid='stMetricValue'] > div {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word;
        line-height: 1.25;
    }}
    [data-baseweb='tab-list'] {{
        gap: 0.45rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--clinverity-border);
    }}
    [data-baseweb='tab'] {{
        background: #f4f8fb;
        border: 1px solid #dbe8f2;
        border-radius: 999px;
        color: var(--clinverity-muted);
        font-weight: 650;
        padding: 0.52rem 0.95rem;
        transition: all 0.18s ease;
    }}
    [data-baseweb='tab']:hover {{
        color: var(--clinverity-primary);
        border-color: rgba(31, 78, 121, 0.28);
        background: #edf5fb;
        box-shadow: var(--clinverity-shadow-soft);
    }}
    [aria-selected='true'][data-baseweb='tab'] {{
        color: #ffffff !important;
        background: var(--clinverity-primary) !important;
        border-color: var(--clinverity-primary) !important;
        box-shadow: 0 14px 28px rgba(31, 78, 121, 0.2);
    }}
    div[data-testid='stExpander'] {{
        background: rgba(243,248,252,0.96);
        border: 1px solid var(--clinverity-border);
        border-radius: 16px;
        overflow: hidden;
        margin-top: 0.45rem;
        margin-bottom: 0.6rem;
        box-shadow: var(--clinverity-shadow-soft);
    }}
    details summary,
    details summary * {{
        background: #f8fbfd;
        color: var(--clinverity-text) !important;
    }}
    [data-testid='stSidebar'] details summary,
    [data-testid='stSidebar'] details summary * {{
        background: #f8fbfd !important;
        color: var(--clinverity-text) !important;
        border: none !important;
    }}
    details summary:hover,
    details summary:hover * {{
        background: #f3f8fc !important;
    }}
    div.stButton, div[data-testid='stDownloadButton'] {{
        display: block;
    }}
    div.stButton > button,
    div[data-testid='stDownloadButton'] > button {{
        border-radius: 12px;
        font-weight: 650;
        color: #ffffff !important;
        background: linear-gradient(180deg, var(--clinverity-primary) 0%, var(--clinverity-primary-dark) 100%) !important;
        border: 1px solid var(--clinverity-primary) !important;
        box-shadow: 0 10px 24px rgba(31, 78, 121, 0.16);
        transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease, border-color 0.16s ease;
        min-height: 2.7rem;
    }}
    div.stButton > button:hover,
    div[data-testid='stDownloadButton'] > button:hover {{
        transform: translateY(-1px);
        background: linear-gradient(180deg, #245a89 0%, #173d60 100%) !important;
        box-shadow: 0 16px 28px rgba(31, 78, 121, 0.18);
    }}
    div.stButton > button:focus,
    div.stButton > button:focus-visible,
    div[data-testid='stDownloadButton'] > button:focus,
    div[data-testid='stDownloadButton'] > button:focus-visible {{
        outline: none;
        box-shadow: 0 0 0 0.22rem rgba(20, 184, 166, 0.28), 0 16px 28px rgba(31, 78, 121, 0.18) !important;
        border-color: var(--clinverity-accent) !important;
    }}
    button:focus,
    button:focus-visible,
    [role='button']:focus,
    [role='button']:focus-visible {{
        outline: 2px solid transparent !important;
        box-shadow: 0 0 0 0.22rem rgba(20, 184, 166, 0.28), 0 12px 24px rgba(31, 78, 121, 0.14) !important;
        border-color: var(--clinverity-accent) !important;
    }}
    div.stButton > button:disabled {{
        opacity: 1 !important;
        background: #bfd1df !important;
        border-color: #bfd1df !important;
        color: #ffffff !important;
        box-shadow: none;
    }}
    div.stButton > button[kind='secondary'],
    div[data-testid='stDownloadButton'] > button[kind='secondary'] {{
        background: #f1f7fb !important;
        color: var(--clinverity-primary) !important;
        border-color: rgba(31, 78, 121, 0.2) !important;
        box-shadow: none;
    }}
    div.stButton > button[kind='secondary']:hover,
    div[data-testid='stDownloadButton'] > button[kind='secondary']:hover {{
        background: var(--clinverity-primary-soft) !important;
        border-color: var(--clinverity-primary) !important;
    }}
    div[data-testid='stDataFrame'],
    div[data-testid='stTable'] {{
        background: var(--clinverity-card);
        border: 1px solid var(--clinverity-border);
        border-radius: var(--clinverity-radius-card);
        box-shadow: var(--clinverity-shadow-soft);
        overflow: hidden;
    }}
    div[data-testid='stDataFrame'] [role='columnheader'],
    div[data-testid='stDataFrame'] [role='gridcell'] {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.28;
    }}
    div[data-testid='stDataFrame'] [role='columnheader'] {{
        background: #f8fbfd;
        color: var(--clinverity-text) !important;
        border-bottom: 1px solid var(--clinverity-border);
        font-weight: 700 !important;
    }}
    .stAlert {{
        border-radius: 16px;
        border: 1px solid var(--clinverity-border);
        box-shadow: var(--clinverity-shadow-soft);
    }}
    [data-baseweb='notification'] {{
        border-radius: 16px !important;
    }}
    .stAlert [data-testid='stMarkdownContainer'] p {{
        line-height: 1.45;
    }}
    .stSelectbox label, .stTextInput label, .stTextArea label, .stNumberInput label, .stDateInput label, .stFileUploader label {{
        color: var(--clinverity-text);
        font-weight: 650;
    }}
    div[data-baseweb='select'] > div,
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stNumberInput input,
    [data-testid='stFileUploaderDropzone'] {{
        border-radius: var(--clinverity-radius-input) !important;
        border-color: var(--clinverity-border) !important;
        background: #f4f8fc !important;
        color: var(--clinverity-text) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }}
    div[data-baseweb='select'] > div:hover,
    .stTextInput > div > div > input:hover,
    .stTextArea textarea:hover,
    .stNumberInput input:hover,
    [data-testid='stFileUploaderDropzone']:hover {{
        border-color: rgba(31, 78, 121, 0.26) !important;
        background: #eef5fb !important;
    }}
    div[data-baseweb='select'] > div:focus-within,
    div[data-baseweb='select'] input:focus,
    div[data-baseweb='select'] input:focus-visible,
    input[role='combobox']:focus,
    input[role='combobox']:focus-visible,
    [role='combobox']:focus,
    [role='combobox']:focus-visible,
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus,
    [data-testid='stFileUploaderDropzone']:focus-within {{
        border-color: var(--clinverity-primary) !important;
        box-shadow: 0 0 0 0.2rem rgba(31, 78, 121, 0.12) !important;
        outline: 2px solid transparent !important;
    }}
    div[data-baseweb='select'] span,
    div[data-baseweb='select'] input,
    div[data-baseweb='select'] div {{
        color: var(--clinverity-text) !important;
    }}
    div[data-baseweb='popover'] {{
        z-index: 9999;
    }}
    body div[data-baseweb='popover'] > div,
    body ul[data-baseweb='menu'],
    body ul[role='listbox'],
    body div[role='listbox'],
    body div[role='dialog'] ul {{
        background: #f4f8fc !important;
        color: var(--clinverity-text) !important;
        border: 1px solid var(--clinverity-border) !important;
        border-radius: 14px !important;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12) !important;
    }}
    body div[data-baseweb='popover'] *,
    body ul[role='listbox'] li,
    body ul[role='listbox'] li *,
    body li[role='option'],
    body li[role='option'] *,
    body div[role='option'],
    body div[role='option'] * {{
        background: transparent !important;
        color: var(--clinverity-text) !important;
    }}
    body li[role='option'],
    body div[role='option'] {{
        border-radius: 10px !important;
        margin: 0.15rem 0.35rem !important;
    }}
    body li[role='option'][aria-selected='true'],
    body li[role='option']:hover,
    body div[role='option'][aria-selected='true'],
    body div[role='option']:hover {{
        background: #eef6fb !important;
        color: var(--clinverity-primary) !important;
    }}
    body li[role='option'][aria-selected='true'] *,
    body li[role='option']:hover *,
    body div[role='option'][aria-selected='true'] *,
    body div[role='option']:hover * {{
        background: #eef6fb !important;
        color: var(--clinverity-primary) !important;
    }}
    [data-testid='stSidebar'] div[data-baseweb='select'] > div {{
        background: #f3f8fc !important;
        color: var(--clinverity-text) !important;
        box-shadow: inset 0 0 0 1px var(--clinverity-border);
    }}
    [data-testid='stSidebar'] .stRadio > div {{
        background: linear-gradient(180deg, #f5f9fd 0%, #eaf2f9 100%);
        border: 1px solid var(--clinverity-border);
        border-radius: 16px;
        padding: 0.45rem 0.55rem;
        box-shadow: var(--clinverity-shadow-soft);
    }}
    [data-testid='stSidebar'] .stRadio [role='radiogroup'] label {{
        border-radius: 12px;
        padding: 0.45rem 0.55rem;
        transition: background 0.16s ease, border-color 0.16s ease;
    }}
    [data-testid='stSidebar'] .stRadio [role='radiogroup'] label:hover {{
        background: var(--clinverity-primary-soft);
    }}
    [data-testid='stSidebar'] .stExpander {{
        margin-bottom: 0.8rem;
    }}
    [data-testid='stFileUploaderDropzone'] {{
        border-style: dashed !important;
        border-width: 1.5px !important;
        background: linear-gradient(180deg, #f5f9fd 0%, #eaf2f9 100%) !important;
        padding: 1rem !important;
    }}
    [data-testid='stFileUploaderDropzone'] [data-testid='stMarkdownContainer'] p {{
        color: var(--clinverity-muted) !important;
    }}
    [data-testid='stFileUploaderDropzone'] small {{
        color: var(--clinverity-muted-soft) !important;
    }}
    [data-testid='stProgressBar'] > div > div {{
        background: linear-gradient(90deg, var(--clinverity-accent) 0%, var(--clinverity-primary) 100%) !important;
        border-radius: 999px !important;
        transition: width 0.28s ease;
    }}
    [data-testid='stProgressBar'] > div {{
        background: #eaf1f6 !important;
        border-radius: 999px !important;
    }}
    .clinverity-panel {{
        background: var(--clinverity-card);
        border: 1px solid var(--clinverity-border);
        border-radius: var(--clinverity-radius-card);
        box-shadow: var(--clinverity-shadow-soft);
        padding: 1rem 1.05rem;
        margin: 0.5rem 0 0.9rem;
    }}
    .clinverity-panel-title {{
        color: var(--clinverity-text);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .clinverity-panel-text {{
        color: var(--clinverity-muted);
        font-size: 0.9rem;
        line-height: 1.52;
    }}
    .clinverity-panel[data-tone='info'] {{
        background: linear-gradient(180deg, #f6fafe 0%, #ebf4fb 100%);
        border-color: rgba(14, 165, 233, 0.18);
    }}
    .clinverity-panel[data-tone='accent'] {{
        background: linear-gradient(180deg, #f4fbfa 0%, #e7f5f2 100%);
        border-color: rgba(20, 184, 166, 0.18);
    }}
    .clinverity-panel[data-tone='warning'] {{
        background: linear-gradient(180deg, #fefaf3 0%, #f7f0df 100%);
        border-color: rgba(245, 158, 11, 0.18);
    }}
    .clinverity-workflow-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.8rem;
        margin: 0.55rem 0 1rem;
    }}
    .clinverity-workflow-step {{
        background: linear-gradient(180deg, #f6fafe 0%, #edf5fb 100%);
        border: 1px solid var(--clinverity-border);
        border-radius: 16px;
        padding: 0.9rem 0.95rem;
        box-shadow: var(--clinverity-shadow-soft);
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    }}
    .clinverity-workflow-step:hover {{
        transform: translateY(-1px);
        border-color: rgba(31, 78, 121, 0.18);
        box-shadow: var(--clinverity-shadow);
    }}
    .clinverity-workflow-step-index {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 1.85rem;
        height: 1.85rem;
        border-radius: 999px;
        background: var(--clinverity-primary-soft);
        color: var(--clinverity-primary);
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }}
    .clinverity-workflow-step-title {{
        color: var(--clinverity-text);
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .clinverity-workflow-step-text {{
        color: var(--clinverity-muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }}
    @keyframes clinverity-fade-in {{
        from {{
            opacity: 0.0;
            transform: translateY(4px);
        }}
        to {{
            opacity: 1.0;
            transform: translateY(0);
        }}
    }}
    [data-testid='stSidebar'] svg {{
        fill: var(--clinverity-muted);
    }}
    .stCheckbox label p,
    .stRadio label p {{
        color: var(--clinverity-text);
    }}
    @media (max-width: 960px) {{
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
        }}
        .clinverity-hero {{
            padding: 1.1rem 1rem 0.95rem;
        }}
        .clinverity-hero-title {{
            font-size: 2rem;
        }}
    }}
</style>
"""

PLOT_LAYOUT = {
    'template': 'plotly',
    'paper_bgcolor': BRAND_COLORS['background_alt'],
    'plot_bgcolor': BRAND_COLORS['card'],
    'font': {'color': BRAND_COLORS['text']},
    'margin': dict(l=30, r=20, t=60, b=30),
}


def _configure_plotly_theme() -> None:
    if pio is None or not hasattr(go, 'layout') or not hasattr(pio, 'templates'):
        return
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=BRAND_COLORS['background_alt'],
        plot_bgcolor=BRAND_COLORS['card'],
        font={'color': BRAND_COLORS['text']},
        colorway=[
            BRAND_COLORS['primary'],
            BRAND_COLORS['accent'],
            '#3B82F6',
            '#F59E0B',
            '#16A34A',
            '#DC2626',
        ],
        xaxis={
            'showgrid': False,
            'zeroline': False,
            'linecolor': '#B8C9D8',
            'gridcolor': '#D5E1EB',
        },
        yaxis={
            'showgrid': True,
            'zeroline': False,
            'linecolor': '#B8C9D8',
            'gridcolor': '#D5E1EB',
        },
    )
    pio.templates['clinverity'] = template
    pio.templates.default = 'clinverity'


def apply_theme() -> None:
    _configure_plotly_theme()
    st.markdown(CARD_STYLE, unsafe_allow_html=True)


def _render_metric_cards(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns([1.25] * len(items), gap='medium')
    for col, (label, value, help_text) in zip(cols, items):
        label_html = escape(label)
        value_html = escape(value)
        help_html = f"<div class='sda-metric-help'>{escape(help_text)}</div>" if help_text else ''
        card_html = (
            "<div class='sda-metric-card'>"
            f"<div class='sda-metric-label'>{label_html}</div>"
            f"<div class='sda-metric-value'>{value_html}</div>"
            f"{help_html}"
            "</div>"
        )
        col.markdown(card_html, unsafe_allow_html=True)


def _normalize_metric_item(item) -> tuple[str, str, str | None]:
    if isinstance(item, dict):
        return str(item.get('label', '')), str(item.get('value', '')), item.get('help')
    if isinstance(item, tuple):
        if len(item) == 2:
            return str(item[0]), str(item[1]), None
        if len(item) >= 3:
            return str(item[0]), str(item[1]), None if item[2] is None else str(item[2])
    return str(item), '', None


def _metric_is_long(label: str, value: str) -> bool:
    return len(label) > 18 or len(value) > 22 or ('/' in value and len(value) > 16)


def _metric_chunks(items: list[tuple[str, str, str | None]]) -> list[list[tuple[str, str, str | None]]]:
    if len(items) <= 3:
        return [items]
    has_long = any(_metric_is_long(label, value) for label, value, _ in items)
    if len(items) == 4 and has_long:
        return [items[:2], items[2:]]
    if len(items) >= 5 and has_long:
        return [items[index:index + 3] for index in range(0, len(items), 3)]
    return [items]


def metric_row(items) -> None:
    normalized = [_normalize_metric_item(item) for item in items]
    for chunk in _metric_chunks(normalized):
        _render_metric_cards(chunk)


def render_app_header(
    *,
    title: str = BRAND_TITLE,
    subtitle: str = BRAND_SUBTITLE,
    tagline: str = BRAND_TAGLINE,
    context_items: list[tuple[str, str]] | None = None,
    build_label: str = '',
) -> None:
    cards_html = ''
    build_html = ''
    if context_items:
        cards = []
        for label, value in context_items:
            cards.append(
                "<div class='clinverity-context-card'>"
                f"<div class='clinverity-context-label'>{escape(str(label))}</div>"
                f"<div class='clinverity-context-value'>{escape(str(value))}</div>"
                "</div>"
            )
        cards_html = f"<div class='clinverity-context-grid'>{''.join(cards)}</div>"
    if str(build_label).strip():
        build_html = f"<div class='clinverity-hero-build'>{escape(build_label)}</div>"
    st.markdown(
        (
            "<section class='clinverity-hero'>"
            "<div class='clinverity-hero-eyebrow'>Clinical data quality platform</div>"
            f"{build_html}"
            f"<div class='clinverity-hero-title'>{escape(title)}</div>"
            f"<div class='clinverity-hero-subtitle'>{escape(subtitle)}</div>"
            f"<div class='clinverity-hero-tagline'>{escape(tagline)}</div>"
            f"{cards_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_sidebar_brand(
    sidebar,
    *,
    workspace_name: str = 'Guest Demo Workspace',
    source_mode: str = 'Demo mode',
    version_note: str = 'Clinverity pilot workspace',
    build_label: str = '',
) -> None:
    build_meta_html = ''
    if str(build_label).strip():
        build_meta_html = (
            "<div class='clinverity-sidebar-meta-item'>"
            "<div class='clinverity-sidebar-meta-label'>Build</div>"
            f"<div class='clinverity-sidebar-meta-value'>{escape(build_label)}</div>"
            "</div>"
        )
    sidebar.markdown(
        (
            "<section class='clinverity-sidebar-brand'>"
            "<div class='clinverity-sidebar-kicker'>Clinverity control panel</div>"
            f"<div class='clinverity-sidebar-title'>{escape(BRAND_TITLE)}</div>"
            f"<div class='clinverity-sidebar-subtitle'>{escape(BRAND_SUBTITLE)}</div>"
            f"<div class='clinverity-sidebar-tagline'>{escape(BRAND_TAGLINE)}</div>"
            "<div class='clinverity-sidebar-meta'>"
            "<div class='clinverity-sidebar-meta-item'>"
            "<div class='clinverity-sidebar-meta-label'>Workspace</div>"
            f"<div class='clinverity-sidebar-meta-value'>{escape(workspace_name)}</div>"
            "</div>"
            "<div class='clinverity-sidebar-meta-item'>"
            "<div class='clinverity-sidebar-meta-label'>Session mode</div>"
            f"<div class='clinverity-sidebar-meta-value'>{escape(source_mode)}</div>"
            "</div>"
            "<div class='clinverity-sidebar-meta-item'>"
            "<div class='clinverity-sidebar-meta-label'>Platform</div>"
            f"<div class='clinverity-sidebar-meta-value'>{escape(version_note)}</div>"
            "</div>"
            f"{build_meta_html}"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_sidebar_section(sidebar, title: str) -> None:
    sidebar.markdown(f"<div class='clinverity-sidebar-section'>{escape(title)}</div>", unsafe_allow_html=True)


def render_sidebar_panel(sidebar, title: str, text: str) -> None:
    sidebar.markdown(
        (
            "<div class='clinverity-sidebar-panel'>"
            f"<div class='clinverity-sidebar-panel-title'>{escape(title)}</div>"
            f"<div class='clinverity-sidebar-panel-text'>{escape(text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_sidebar_status_panel(sidebar, *, dataset_name: str, source_mode: str, row_count: str, column_count: str) -> None:
    sidebar.markdown(
        (
            "<div class='clinverity-sidebar-panel'>"
            "<div class='clinverity-sidebar-panel-title'>Active dataset</div>"
            "<div class='clinverity-sidebar-stat-grid'>"
            "<div class='clinverity-sidebar-stat'>"
            "<div class='clinverity-sidebar-stat-label'>Dataset</div>"
            f"<div class='clinverity-sidebar-stat-value'>{escape(dataset_name)}</div>"
            "</div>"
            "<div class='clinverity-sidebar-stat'>"
            "<div class='clinverity-sidebar-stat-label'>Source</div>"
            f"<div class='clinverity-sidebar-stat-value'>{escape(source_mode)}</div>"
            "</div>"
            "<div class='clinverity-sidebar-stat'>"
            "<div class='clinverity-sidebar-stat-label'>Shape</div>"
            f"<div class='clinverity-sidebar-stat-value'>{escape(row_count)} rows · {escape(column_count)} columns</div>"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, description: str | None = None) -> None:
    description_html = (
        f"<div class='clinverity-section-description'>{escape(description)}</div>"
        if description
        else ''
    )
    st.markdown(
        (
            "<div class='clinverity-section-intro'>"
            f"<div class='clinverity-section-title'>{escape(title)}</div>"
            f"{description_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_subsection_header(title: str, note: str | None = None) -> None:
    note_html = f"<div class='clinverity-subsection-note'>{escape(note)}</div>" if note else ''
    st.markdown(
        (
            "<div class='clinverity-subsection'>"
            "<div class='clinverity-subsection-copy'>"
            f"<div class='clinverity-subsection-title'>{escape(title)}</div>"
            f"{note_html}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_badge_row(items: list[tuple[str, str]] | list[str]) -> None:
    badges: list[str] = []
    for item in items:
        if isinstance(item, tuple):
            label, tone = item
        else:
            label, tone = str(item), 'default'
        badges.append(f"<span class='clinverity-badge' data-tone='{escape(str(tone))}'>{escape(str(label))}</span>")
    if badges:
        st.markdown(f"<div class='clinverity-badge-row'>{''.join(badges)}</div>", unsafe_allow_html=True)


def render_surface_panel(title: str, text: str, *, tone: str = 'default') -> None:
    st.markdown(
        (
            f"<div class='clinverity-panel' data-tone='{escape(tone)}'>"
            f"<div class='clinverity-panel-title'>{escape(title)}</div>"
            f"<div class='clinverity-panel-text'>{escape(text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_workflow_steps(steps: list[tuple[str, str]]) -> None:
    cards = []
    for index, (title, text) in enumerate(steps, start=1):
        cards.append(
            "<div class='clinverity-workflow-step'>"
            f"<div class='clinverity-workflow-step-index'>{index}</div>"
            f"<div class='clinverity-workflow-step-title'>{escape(title)}</div>"
            f"<div class='clinverity-workflow-step-text'>{escape(text)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='clinverity-workflow-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_advanced_sections_toggle(
    section_key: str,
    *,
    label: str = 'Show advanced sections by default',
    help_text: str | None = None,
) -> bool:
    session_key = f'{section_key}_advanced_sections_enabled'
    role = str(st.session_state.get('workspace_role') or st.session_state.get('active_role') or '').strip()
    if session_key not in st.session_state:
        st.session_state[session_key] = role in {'Admin', 'Data Steward', 'Owner'}
    return bool(
        st.toggle(
            label,
            key=session_key,
            help=help_text or 'Keep detailed admin, governance, and audit sections expanded by default for the current workspace view.',
        )
    )


def render_role_context_panel(
    role: str,
    *,
    primary_message: str,
    advanced_enabled: bool,
    advanced_label: str = 'Advanced sections',
) -> None:
    render_surface_panel(
        f'{role} view mode',
        (
            f"{primary_message} "
            f"{advanced_label} are currently {'expanded' if advanced_enabled else 'collapsed'} by default."
        ),
        tone='info' if advanced_enabled else 'default',
    )


def style_figure(fig, xaxis_title: str | None = None, yaxis_title: str | None = None):
    if fig is None or not hasattr(fig, 'update_layout'):
        return fig
    fig.update_layout(**PLOT_LAYOUT, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BRAND_COLORS['border'])
    fig.update_yaxes(gridcolor='#E8EEF5', zeroline=False, linecolor=BRAND_COLORS['border'])
    return fig


def plot_missingness(missing_df: pd.DataFrame):
    if missing_df.empty:
        return None
    figure = px.bar(
        missing_df.head(15),
        x='column_name',
        y='null_percentage',
        title='Highest Missingness by Column',
        color='null_percentage',
        color_continuous_scale=['#DCEAF6', BRAND_COLORS['primary'], BRAND_COLORS['error']],
    )
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, 'Column', 'Null Percentage')


def plot_numeric_distribution(data: pd.DataFrame, column: str):
    if column not in data.columns:
        return None
    series = pd.to_numeric(data[column], errors='coerce').dropna()
    if len(series) > 5000:
        series = series.sample(5000, random_state=42)
    if series.empty:
        return None
    figure = px.histogram(series.to_frame(name=column), x=column, nbins=30, title=f'Distribution of {column}')
    figure.update_traces(marker_color=BRAND_COLORS['primary'])
    return style_figure(figure, column, 'Rows')


def plot_top_categories(data: pd.DataFrame, column: str):
    if column not in data.columns:
        return None
    summary = data[column].fillna('Missing').astype(str).value_counts().head(12).rename_axis(column).reset_index(name='count')
    if summary.empty:
        return None
    figure = px.bar(
        summary,
        x=column,
        y='count',
        title=f'Top Values for {column}',
        color='count',
        color_continuous_scale=['#D8F3F0', BRAND_COLORS['accent'], BRAND_COLORS['primary']],
    )
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, column, 'Rows')


def plot_correlation(corr: pd.DataFrame):
    if corr.empty:
        return None
    figure = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale=['#DC2626', BRAND_COLORS['background_alt'], '#14B8A6'],
        title='Correlation Heatmap',
    )
    return style_figure(figure, 'Numeric Field', 'Numeric Field')


def plot_time_trend(trend_df: pd.DataFrame, x_col: str, y_col: str, title: str):
    if trend_df.empty or y_col not in trend_df.columns:
        return None
    figure = px.line(trend_df, x=x_col, y=y_col, markers=True, title=title)
    figure.update_traces(line_color=BRAND_COLORS['primary'], marker_color=BRAND_COLORS['accent'])
    return style_figure(figure, x_col.replace('_', ' ').title(), y_col.replace('_', ' ').title())


def plot_bar(data: pd.DataFrame, x_col: str, y_col: str, title: str):
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return None
    figure = px.bar(
        data.head(15),
        x=x_col,
        y=y_col,
        title=title,
        color=y_col,
        color_continuous_scale=['#DCEAF6', BRAND_COLORS['accent'], BRAND_COLORS['primary']],
    )
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, x_col.replace('_', ' ').title(), y_col.replace('_', ' ').title())


def plot_numeric_box(data: pd.DataFrame, column: str):
    if column not in data.columns:
        return None
    series = pd.to_numeric(data[column], errors='coerce').dropna()
    if len(series) > 5000:
        series = series.sample(5000, random_state=42)
    if series.empty:
        return None
    figure = px.box(series.to_frame(name=column), y=column, title=f'Box Plot for {column}')
    figure.update_traces(marker_color=BRAND_COLORS['primary'], line_color=BRAND_COLORS['accent'])
    return style_figure(figure, None, column)
