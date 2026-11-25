"""CSS and small HTML helpers that make the Streamlit UI look nicer."""
from __future__ import annotations

import html

CSS = """
<style>
.block-container {padding-top: 3.2rem; max-width: 1200px;}
.hero {
  background: linear-gradient(135deg, #4338CA 0%, #6D28D9 55%, #A855F7 100%);
  color: #fff; padding: 1.4rem 1.8rem; border-radius: 18px; margin-bottom: 1.1rem;
  box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
}
.hero h1 {color: #fff; font-size: 1.7rem; margin: 0 0 .25rem 0; padding: 0;}
.hero p {color: rgba(255,255,255,.88); margin: 0; font-size: .95rem;}
.brand {font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em;}
.brand span {background: linear-gradient(90deg,#4F46E5,#A855F7);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem;
  font-weight:600; margin:0 6px 4px 0; border:1px solid transparent;}
.b-score {background:#FEF3C7; color:#92400E;}
.b-ieee {background:#DBEAFE; color:#1E40AF;}
.b-oa {background:#DCFCE7; color:#166534;}
.b-lock {background:#F1F5F9; color:#475569;}
.b-cite {background:#EDE9FE; color:#5B21B6;}
.b-done {background:#D1FAE5; color:#065F46;}
.b-abs {background:#FFEDD5; color:#9A3412;}
.b-up {background:#FCE7F3; color:#9D174D;}
.ptitle {font-weight:700; font-size:1.02rem; line-height:1.35; margin-bottom:.2rem;}
.pmeta {color:#64748B; font-size:.83rem; margin-bottom:.35rem;}
.preason {color:#4338CA; font-size:.86rem; font-style:italic;}
.field-label {font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:#6D28D9; margin-top:.4rem;}
.field-text {font-size:.9rem; line-height:1.5; margin-bottom:.3rem;}
.empty {text-align:center; padding:2.2rem 1rem; color:#64748B; border:2px dashed #D9DCEB;
  border-radius:16px; background:#fff;}
.empty b {color:#1E1B3A; font-size:1.05rem;}
[data-testid="stMetric"] {background:#fff; border:1px solid #E6E8F2; border-radius:14px;
  padding:12px 16px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background:#fff; border-radius:10px 10px 0 0; padding:8px 16px;}
.st-key-synopsis_page {background:#fff; padding:2rem 2.6rem; border-radius:14px;
  box-shadow:0 6px 28px rgba(30,27,58,.08); border:1px solid #ECEDF5;}
.st-key-synopsis_page h1 {font-size:1.45rem; text-align:center; color:#3730A3;}
.st-key-synopsis_page h2 {font-size:1.02rem; color:#3730A3; margin-top:.6rem;
  border-bottom:1px solid #EEF0F8; padding-bottom:.15rem;}
.st-key-synopsis_page p, .st-key-synopsis_page li {font-size:.93rem; line-height:1.55;
  text-align:justify;}
</style>
"""


def badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{html.escape(text)}</span>'


def hero(title: str, subtitle: str) -> str:
    return (f'<div class="hero"><h1>{html.escape(title)}</h1>'
            f'<p>{html.escape(subtitle)}</p></div>')


def empty_state(title: str, text: str) -> str:
    return f'<div class="empty"><b>{html.escape(title)}</b><br>{html.escape(text)}</div>'


def field(label: str, text: str) -> str:
    return (f'<div class="field-label">{html.escape(label)}</div>'
            f'<div class="field-text">{html.escape(text)}</div>')
