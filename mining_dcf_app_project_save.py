"""
Mining Project DCF Valuation
Run:     streamlit run mining_dcf_app.py
Install: pip install streamlit pandas numpy matplotlib scipy openpyxl
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy.optimize import brentq
from copy import deepcopy
import io, warnings, json
from datetime import datetime
warnings.filterwarnings("ignore")


def _idx(options, value, fallback=0):
    try:
        return options.index(value)
    except Exception:
        return fallback

def _load_project_payload(uploaded_file):
    try:
        payload = json.load(uploaded_file)
        if isinstance(payload, dict) and isinstance(payload.get("inputs"), dict):
            return payload["inputs"], payload
        if isinstance(payload, dict):
            return payload, {"inputs": payload}
    except Exception:
        return None, None
    return None, None

def _apply_project_inputs(inputs):
    for k, v in inputs.items():
        st.session_state[k] = v if v != 0.0 else st.session_state.get(k, v)
    for k in ["res", "mc"]:
        if k in st.session_state:
            del st.session_state[k]


st.set_page_config(page_title="Mining DCF", page_icon="⛏", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'Source Sans 3',sans-serif; }
.header { background:linear-gradient(135deg,#0a1628 0%,#0f3460 100%);
    padding:22px 32px 18px; border-radius:10px; margin-bottom:20px; border-left:5px solid #c9a84c; }
.header h1 { font-family:'Libre Baskerville',serif; color:#fff; font-size:1.75em; margin:0 0 3px; }
.header p  { color:#aab7c4; font-size:.85em; margin:0; }
.kcard { background:white; border-radius:8px; padding:13px 16px;
         border-top:3px solid #0f3460; box-shadow:0 2px 6px rgba(0,0,0,.06); text-align:center; }
.klbl  { font-size:.67em; color:#7f8c8d; text-transform:uppercase; letter-spacing:.7px; font-weight:600; }
.kval  { font-size:1.6em; font-weight:700; font-family:'Libre Baskerville',serif; margin-top:3px; }
.pos{color:#1e8449;} .neg{color:#c0392b;} .neu{color:#0f3460;}
.shdr  { font-family:'Libre Baskerville',serif; font-size:.95em; color:#0f3460;
         border-bottom:2px solid #c9a84c; padding-bottom:4px; margin:16px 0 8px; font-weight:700; }
.note  { background:#eaf4fb; border-left:3px solid #2471a3; padding:7px 10px;
         border-radius:0 5px 5px 0; font-size:.82em; color:#1a5276; margin:5px 0 10px; }
.warn  { background:#fef9ec; border-left:3px solid #c9a84c; padding:7px 10px;
         border-radius:0 5px 5px 0; font-size:.82em; color:#7d5a00; margin:5px 0 10px; }
.err   { background:#fdecea; border-left:3px solid #c0392b; padding:8px 12px;
         border-radius:5px; font-size:.85em; color:#922b21; margin:6px 0; }
.foot  { font-size:.73em; color:#95a5a6; margin-top:6px; }
[data-testid="stSidebar"] { background:#0d1f3c; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stTextInput label { color:#e8f0f8 !important; font-size:.84em; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color:#ffffff !important; font-size:.86em !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap:6px; }
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] { background:#1a2f4e !important; }
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span { color:#ffffff !important; }
[data-testid="stSidebar"] .stSelectbox svg { fill:#e8f0f8 !important; }
[data-testid="stSidebar"] p { color:#e8f0f8 !important; }
[data-testid="stSidebar"] input { color:#ffffff !important; }
[data-testid="stSidebar"] .stNumberInput input { color:#ffffff !important; background:#1a2f4e !important; }
[data-testid="stSidebar"] .stTextInput input { color:#ffffff !important; background:#1a2f4e !important; }
[data-testid="stSidebar"] .note { background:#0f2540 !important; color:#a8d4f5 !important; border-left-color:#3a7fc1 !important; }
[data-testid="stSidebar"] .warn { background:#1a2a0a !important; color:#c8e090 !important; border-left-color:#7aaa20 !important; }
[data-testid="stSidebar"] .stMarkdown h3 { color:#ffffff !important;
    font-family:'Libre Baskerville',serif !important; font-size:1.0em !important;
    font-weight:700; margin:0 0 12px 0 !important; letter-spacing:.3px; }
[data-testid="stSidebar"] .stMarkdown h4 { color:#c9a84c !important;
    font-family:'Libre Baskerville',serif !important; font-size:.82em !important;
    text-transform:uppercase; letter-spacing:.5px; margin:14px 0 4px !important;
    border-bottom:1px solid #2a4a6e; padding-bottom:3px; }

/* ── Expander (Project File) ── */
[data-testid="stSidebar"] details summary,
[data-testid="stSidebar"] details summary * {
    color: #7a9bbf !important;
    background-color: #0d1f3c !important;
    transition: color 0.2s ease, background-color 0.2s ease !important;
}
[data-testid="stSidebar"] details summary:hover,
[data-testid="stSidebar"] details summary:focus,
[data-testid="stSidebar"] details summary:hover *,
[data-testid="stSidebar"] details summary:focus * {
    color: #ffffff !important;
    background-color: #1a3459 !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] details[open] summary,
[data-testid="stSidebar"] details[open] summary *,
[data-testid="stSidebar"] details[open] {
    background-color: #0d1f3c !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #0d1f3c !important;
    border: 1px solid #2a4a6e !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background-color: #0d1f3c !important;
}

/* ── File uploader section ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] div { color:#f3f6fb !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:#102646 !important; border:1px solid #35577e !important; }

/* ── Dropzone ── */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: #1a2f4e !important;
    border: 1px dashed #4f6f9b !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] label,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] div {
    color: #dbe7f3 !important;
    opacity: 1 !important;
}

/* ── General sidebar buttons (Load Project etc.) ── */
[data-testid="stSidebar"] .stButton > button {
    color: #7a9bbf !important;
    background-color: #0d1f3c !important;
    border: 1px solid #2a4a6e !important;
    transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    color: #ffffff !important;
    background-color: #1a3459 !important;
    border-color: #5d7ea8 !important;
}

/* ── Browse Files button ── */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] ~ div button,
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    color: #7a9bbf !important;
    background-color: #0d1f3c !important;
    border: 1px solid #2a4a6e !important;
    transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] ~ div button:hover,
[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    color: #ffffff !important;
    background-color: #1a3459 !important;
    border-color: #5d7ea8 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMMODITY CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GRADE_SYSTEM = {
    "Gold":"g/t","Silver":"g/t",
    "Copper":"pct","Iron Ore":"pct","Lithium":"pct",
    "Nickel":"pct","Zinc":"pct","Coal":"pct","Other":"pct",
}
PROD_UNITS = {
    "Gold":["oz","koz","g","kg","t"],"Silver":["oz","koz","g","kg","t"],
    "Copper":["lb","Mlb","t","kt"],"Iron Ore":["t","kt","Mt"],
    "Lithium":["t","kt"],"Nickel":["t","kt"],
    "Zinc":["lb","t","kt"],"Coal":["t","kt","Mt"],
    "Other":["t","kt","oz","lb","kg"],
}
PRICE_UNITS = {
    "Gold":["$/oz","$/g","$/kg","$/t"],"Silver":["$/oz","$/g","$/kg","$/t"],
    "Copper":["$/lb","$/t"],"Iron Ore":["$/t","$/dmt"],
    "Lithium":["$/t","$/kg"],"Nickel":["$/t","$/lb"],
    "Zinc":["$/lb","$/t"],"Coal":["$/t"],
    "Other":["$/oz","$/t","$/lb","$/kg"],
}
DEFAULTS = {
    # price  pstep  grade   rec   opex_t  opex_pu  opex_step  payable
    "Gold":    (2000., 10.,  1.5,  90., 35.,  900.,  10.,  99.),
    "Silver":  (25.,   0.5,  120., 88., 30.,  12.,   0.5,  99.),
    "Copper":  (3.5,   0.05, 2.45, 87., 53.,  1.62,  0.01, 96.5),
    "Iron Ore":(120.,  1.,   62.,  90., 8.,   25.,   1.,   98.),
    "Lithium": (15000.,100., 1.2,  78., 45.,  4000., 100., 97.),
    "Nickel":  (18000.,100., 1.5,  83., 60.,  8000., 100., 97.),
    "Zinc":    (1.2,   0.05, 8.,   85., 40.,  0.6,   0.01, 85.),
    "Coal":    (180.,  5.,   15.,  85., 55.,  60.,   1.,   98.),
    "Other":   (100.,  1.,   1.,   85., 30.,  30.,   1.,   95.),
}
PROD_TO_T = {
    "oz":1/32150.7,"koz":1e3/32150.7,"g":1e-6,"kg":1e-3,
    "t":1.,"kt":1e3,"Mt":1e6,"lb":1/2204.62,"Mlb":1e6/2204.62,
}
PRICE_TO_T = {
    "$/oz":1/32150.7,"$/g":1e-6,"$/kg":1e-3,"$/t":1.,
    "$/dmt":1.,"$/lb":1/2204.62,
}

def unit_conv(prod_u, price_u):
    return PROD_TO_T.get(prod_u,1.) / PRICE_TO_T.get(price_u,1.)

def grade_to_contained_t(tpa_ore, grade_val, grade_sys):
    """ore tonnes + grade → tonnes of contained metal"""
    if grade_sys == "g/t":
        return tpa_ore * grade_val / 1_000_000
    return tpa_ore * grade_val / 100.0


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def build_schedule(p):
    ml  = int(p["mine_life"])
    cy_raw = float(p.get("construction_years", 0))
    cy  = int(round(cy_raw))  # whole years for the schedule loop
    # CAPEX is spread over the fractional construction period
    # For sub-year construction (e.g. 0.5), cy=0 so capex hits Year 1 pre-production
    # We handle this by treating cy<1 as 0 construction years (capex in Year 1)
    dr  = float(p["discount_rate"]) / 100
    tr  = float(p["tax_rate"])       / 100
    rr  = float(p["royalty_rate"])   / 100
    pe  = float(p.get("price_esc", 0))  / 100
    oe  = float(p.get("opex_esc",  0))  / 100
    ramp= float(p.get("ramp_up",  100)) / 100
    pay = float(p.get("payable",  100)) / 100
    ic  = float(p["initial_capex"]) * 1e6
    sc  = float(p.get("sust_capex", 0)) * 1e6
    cl  = float(p.get("closure",    0)) * 1e6
    uc  = float(p.get("unit_conv",  1.))
    byp_pa = float(p.get("byp_rev", 0)) * 1e6

    # ── Derive annual production and unit opex from input mode ────────────────
    mode = p.get("input_mode", "summary")

    if mode == "ore":
        # Ore-based: throughput × grade × recovery
        tpa = float(p["throughput_tpd"]) * 365 * float(p.get("avail_pct", 91)) / 100
        contained_t = grade_to_contained_t(tpa, float(p["grade_value"]), p["grade_system"])
        rec_t = contained_t * float(p["recovery_pct"]) / 100
        # recovered tonnes → production unit
        ann_base = rec_t / PROD_TO_T.get(p.get("unit","t"), 1.)
        ann_pu = ann_base * uc
        opex_tpa = float(p.get("opex_t", 0))
        opex_pu_direct = float(p.get("opex_pu", 0))
        if opex_tpa > 0:
            # $/t processed → $/price-unit
            opex_base = (opex_tpa * tpa / ann_pu) if ann_pu > 0 else 0.
        else:
            # $/price-unit entered directly (e.g. C1 in $/oz)
            opex_base = opex_pu_direct
        # by-product from grade
        if float(p.get("byp_grade", 0)) > 0:
            byp_oz = tpa * float(p["byp_grade"]) * float(p.get("byp_rec", 75)) / 100 / 31.1035
            byp_pa = byp_oz * float(p.get("byp_price", 0)) * float(p.get("byp_pay", 90)) / 100
    else:
        # Summary / direct: annual production entered directly
        ann_base  = float(p["annual_prod"])
        opex_base = float(p["opex_pu"])  # already in $/price-unit
        tpa = 0.

    # ── Depreciation pool ─────────────────────────────────────────────────────
    # Initial capex: straight-line over mine life
    dep_ic = ic / ml if ml > 0 else 0.
    # Sustaining capex: immediately deductible (operating in nature)
    # This is labelled clearly in the output

    records, loss, wc = [], 0., 0.
    for yr in range(1, ml + cy + 1):
        op = yr - cy

        # CAPEX spend
        capex_yr = (ic/cy) if (cy > 0 and op <= 0) else (ic if yr==1 and cy==0 else 0.)
        sust_yr  = sc if op > 0 else 0.
        cl_yr    = cl if op == ml else 0.

        # Production
        if   op <= 0: prod = 0.
        elif op == 1: prod = ann_base * ramp
        else:         prod = ann_base

        esc    = max(op - 1, 0)
        price  = float(p["commodity_price"]) * (1+pe)**esc if op > 0 else 0.
        opex_u = opex_base * (1+oe)**esc                   if op > 0 else 0.

        # Revenue
        gross = prod * uc * price * pay
        total = gross + (byp_pa if op > 0 else 0.)
        royal = gross * rr
        opex_tot = prod * uc * opex_u

        # EBITDA — sustaining capex is treated as immediately deductible for tax
        # (cash cost in the period, consistent with how most junior studies treat it)
        ebitda = total - opex_tot - royal

        # Taxable income: EBITDA less initial-capex depreciation less sust capex
        dep_yr = dep_ic if op > 0 else 0.
        taxable = (ebitda - dep_yr - sust_yr) - loss
        tax_yr  = max(taxable * tr, 0.)
        # Carry forward any losses
        if taxable < 0:
            loss = abs(taxable)
        else:
            loss = max(loss - (ebitda - dep_yr - sust_yr), 0.)

        # Working capital: 25% of Year-1 opex, recovered in final year
        wc_yr = 0.
        if op == 1:    wc_yr = -(opex_tot * 0.25); wc = -wc_yr
        elif op == ml: wc_yr = wc

        # FCF: EBITDA minus tax, capex, working capital
        # Sustaining capex already in taxable income; subtract from FCF here
        fcf = ebitda - tax_yr - capex_yr - sust_yr - cl_yr + wc_yr
        df_ = 1/(1+dr)**((op-0.5) if op > 0 else (yr-0.5))

        records.append({
            "Year":f"Yr {yr}", "Op.Yr":op, "Production":prod,
            "Price":price, "Revenue":total, "OPEX":opex_tot,
            "Royalties":royal, "EBITDA":ebitda,
            "Depreciation":dep_yr, "Tax":tax_yr,
            "Init.Capex":capex_yr, "Sust.Capex":sust_yr,
            "FCF":fcf, "DF":df_, "PV":fcf*df_,
        })

    df = pd.DataFrame(records)
    df["CumFCF"] = df["FCF"].cumsum()
    df["CumPV"]  = df["PV"].cumsum()
    return df, tpa, ann_base, opex_base


def kpis(df, p):
    ic   = float(p["initial_capex"]) * 1e6
    npv  = df["PV"].sum()
    fcfs = df["FCF"].values; n = len(fcfs)
    # For IRR: capex should be the t=0 outflow regardless of construction structure
    # When cy=0, capex is in year 1 FCF — extract it to t=0 for a proper IRR calc
    ic_irr = float(p.get("initial_capex", 0)) * 1e6
    cy_irr = int(round(float(p.get("construction_years", 0))))
    if cy_irr == 0:
        # Separate capex from year-1 FCF and place at t=0
        fcfs_irr = np.concatenate([[-ic_irr], fcfs.copy()])
        fcfs_irr[1] = fcfs_irr[1] + ic_irr  # add capex back to yr-1 FCF
        n_irr = len(fcfs_irr)
        try:    irr = brentq(lambda r: np.sum(fcfs_irr/(1+r)**(np.arange(n_irr))), -.9999, 20, maxiter=1000)*100
        except: irr = float("nan")
    else:
        try:    irr = brentq(lambda r: np.sum(fcfs/(1+r)**(np.arange(n)+0.5)), -.9999, 20, maxiter=1000)*100
        except: irr = float("nan")

    # Payback: pre-accumulate construction then find operating crossover
    cum = 0.
    for _, row in df.iterrows():
        if row["Op.Yr"] <= 0:
            cum += row["FCF"]
    pb = float("nan")
    for _, row in df.iterrows():
        if row["Op.Yr"] > 0:
            prev = cum; cum += row["FCF"]
            if prev < 0 <= cum and prev != cum:
                pb = (row["Op.Yr"]-1) + (-prev/(cum-prev)); break

    op   = df[df["Op.Yr"]>0]
    tp   = op["Production"].sum()
    tr_  = op["Revenue"].sum()
    to   = op["OPEX"].sum()
    tro  = op["Royalties"].sum()
    ts   = op["Sust.Capex"].sum()
    tt   = op["Tax"].sum()
    uc   = float(p.get("unit_conv",1.))
    tp_pu = tp * uc  # total production in price units

    # Pre-tax NPV: discount FCF before tax (add back tax each year)
    df_pretax = df.copy()
    df_pretax["FCF_pretax"] = df_pretax["FCF"] + df_pretax["Tax"]
    npv_pretax = (df_pretax["FCF_pretax"] * df_pretax["DF"]).sum()
    
    tebitda = op["EBITDA"].sum()
    ann_ebitda = tebitda / float(p.get("mine_life", 1))

    return {
        "npv":npv/1e6, "npv_pretax":npv_pretax/1e6, "irr":irr, "pb":pb,
        "aisc":(to+ts+tro)/tp_pu if tp_pu>0 else 0.,
        "c1":(to+tro)/tp_pu       if tp_pu>0 else 0.,
        "eb":op["EBITDA"].sum()/tr_*100 if tr_>0 else 0.,
        "mult":npv/ic if ic>0 else float("nan"),
        "tp":tp, "tr":tr_/1e6, "to":to/1e6, "tt":tt/1e6,
        "tebitda":tebitda/1e6, "ann_ebitda":ann_ebitda/1e6,
        "trev":tr_/1e6, "tfcf":df["FCF"].sum()/1e6,
    }


if "project_name" not in st.session_state:
    st.session_state["project_name"] = "My Project"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    with st.expander("💾 Project file", expanded=False):
        uploaded_project = st.file_uploader("Load saved project (.json)", type=["json"], key="project_file_upload")
        loaded_project_inputs = None
        loaded_project_meta = None
        if uploaded_project is not None:
            loaded_project_inputs, loaded_project_meta = _load_project_payload(uploaded_project)
            if loaded_project_inputs is None:
                st.markdown('<div class="err">Could not read project file.</div>', unsafe_allow_html=True)
            else:
                project_label = loaded_project_meta.get("project_name") or loaded_project_inputs.get("project_name") or uploaded_project.name
                st.markdown(f'<div class="note">Loaded file ready: <b>{project_label}</b></div>', unsafe_allow_html=True)
        if st.button("Load project into inputs", key="load_project_btn", use_container_width=True, disabled=loaded_project_inputs is None):
            _apply_project_inputs(loaded_project_inputs)
            st.rerun()

    st.markdown("### ⛏ Project Inputs")

    # ── 1. Project ────────────────────────────────────────────────────────────
    st.markdown("#### Project")
    name      = st.text_input("Project name", key="project_name")
    commodity = st.selectbox("Commodity",
                    ["Gold","Copper","Silver","Iron Ore","Lithium","Nickel","Zinc","Coal","Other"],
                    index=_idx(["Gold","Copper","Silver","Iron Ore","Lithium","Nickel","Zinc","Coal","Other"], st.session_state.get("commodity", "Gold")), key="commodity")
    d = DEFAULTS[commodity]

    c1, c2 = st.columns(2)
    unit       = c1.selectbox("Production unit", PROD_UNITS[commodity], index=_idx(PROD_UNITS[commodity], st.session_state.get("unit", PROD_UNITS[commodity][0])), key="unit")
    price_unit = c2.selectbox("Price / cost unit", PRICE_UNITS[commodity], index=_idx(PRICE_UNITS[commodity], st.session_state.get("price_unit", PRICE_UNITS[commodity][0])), key="price_unit")
    pu  = price_unit
    pul = pu.replace("$/","")   # e.g. "oz"
    uc  = unit_conv(unit, price_unit)
    grade_sys = GRADE_SYSTEM[commodity]

    # ── 2. Input mode ─────────────────────────────────────────────────────────
    st.markdown("#### Input mode")
    input_mode = st.radio(
        "How are you entering production?",
        ["Study summary  (annual production direct)",
         "Ore-based  (throughput × grade × recovery)"],
        index=_idx(["Study summary  (annual production direct)", "Ore-based  (throughput × grade × recovery)"], st.session_state.get("input_mode_label", "Study summary  (annual production direct)")),
        key="input_mode_label",
        label_visibility="collapsed")
    use_ore = input_mode.startswith("Ore")

    # ── 3. Mine schedule ──────────────────────────────────────────────────────
    st.markdown("#### Mine schedule")
    c1, c2 = st.columns(2)
    mine_life   = c1.number_input("Mine life  (years)",    1, 100, int(st.session_state.get("mine_life", 10)), 1, key="mine_life")
    const_years = c2.number_input("Construction  (years)", 0.0, 20.0, float(st.session_state.get("construction_years", 2.0)), 0.5, format="%.1f", key="construction_years",
        help="Time to build before first production, in years. Enter 0.5 for 6 months, 1.5 for 18 months etc.")
    ramp_up     = st.number_input("Year 1 production  (% of full rate)", 0.0, 100.0, float(st.session_state.get("ramp_up", 80.0)), 0.5, format="%.1f", key="ramp_up",
        help="Year 1 production as % of steady-state annual rate.\ne.g. 50% = mine reaches full capacity halfway through Year 1.\n100% = full production from day one.")

    # ── 4. Production ─────────────────────────────────────────────────────────
    st.markdown("#### Production")

    if use_ore:
        # Throughput — accept total LOM tonnes or annual or tpd
        grade_lbl = "g/t" if grade_sys == "g/t" else "%"
        ore_input_type = st.radio(
            "Ore tonnes entered as",
            ["Total LOM  (e.g. 1.2 Mt resource)", "Annual  (Mt/yr)", "Daily  (tpd)"],
            index=_idx(["Total LOM  (e.g. 1.2 Mt resource)", "Annual  (Mt/yr)", "Daily  (tpd)"], st.session_state.get("ore_input_type", "Total LOM  (e.g. 1.2 Mt resource)")), key="ore_input_type",
            horizontal=True, label_visibility="collapsed",
            help="Choose how your study quotes ore tonnage.\n"
                 "Total LOM: e.g. '1.2 Mt @ 2.22 g/t' from a resource statement\n"
                 "Annual: e.g. '0.6 Mtpa processed'\n"
                 "Daily: e.g. '1,644 tpd plant capacity'")

        c1, c2 = st.columns(2)
        if ore_input_type.startswith("Total"):
            total_ore_mt = c1.number_input("Total ore  (Mt)", 0.001, 10000., float(st.session_state.get("total_ore_mt", 1.2)), 0.1, key="total_ore_mt",
                format="%.3f",
                help="Total LOM ore tonnes in millions — the number quoted alongside grade\n"
                     "e.g. '1.2 Mt @ 2.22 g/t' → enter 1.2")
            avail = c2.number_input("Mill availability  (%)", 50., 100., float(st.session_state.get("avail_pct", 91.0)), 0.5, key="avail_pct")
            tpa   = total_ore_mt * 1e6 / mine_life
            tpd   = tpa / (365 * avail / 100)
        elif ore_input_type.startswith("Annual"):
            annual_mt = c1.number_input("Annual throughput  (Mt/yr)", 0.001, 1000., float(st.session_state.get("annual_mt", 0.6)), 0.05, key="annual_mt",
                format="%.3f",
                help="Ore processed per year in million tonnes\n"
                     "e.g. '0.6 Mtpa' → enter 0.6")
            avail = c2.number_input("Mill availability  (%)", 50., 100., float(st.session_state.get("avail_pct", 91.0)), 0.5, key="avail_pct")
            tpa   = annual_mt * 1e6
            tpd   = tpa / (365 * avail / 100)
        else:
            tpd   = c1.number_input("Throughput  (tpd)", 1., 500000., max(1.0, float(st.session_state.get("throughput_tpd", 1644.0))), 10., key="throughput_tpd",
                help="Plant capacity in tonnes of ore per day\n"
                     "Divide annual by (365 × availability) to get tpd")
            avail = c2.number_input("Mill availability  (%)", 50., 100., float(st.session_state.get("avail_pct", 91.0)), 0.5, key="avail_pct")
            tpa   = tpd * 365 * avail / 100

        c1, c2 = st.columns(2)
        grade_val = c1.number_input(f"Head grade  ({grade_lbl})",
            0.001, 100000. if grade_sys=="g/t" else 100.,
            float(st.session_state.get("grade_value", float(d[2]))), 0.01, format="%.3f", key="grade_value",
            help=f"Average grade of ore fed to the plant in {grade_lbl}\n"
                 f"This is the mill feed / processed grade — not the resource grade")
        recov = c2.number_input("Recovery  (%)", 1., 100., float(st.session_state.get("recovery_pct", float(d[3]))), 0.5, key="recovery_pct",
            help="Metallurgical recovery — % of contained metal extracted into final product")

        # Preview
        cont_t = grade_to_contained_t(tpa, grade_val, grade_sys)
        rec_t  = cont_t * recov / 100
        ann    = rec_t / PROD_TO_T.get(unit, 1.)
        ann_pu = ann * uc
        if grade_sys == "g/t":
            formula = f"{tpa/1e6:.3f} Mt/yr × {grade_val:.3f} g/t ÷ 31.1035 × {recov:.0f}%"
        else:
            formula = f"{tpa/1e6:.3f} Mt/yr × {grade_val:.3f}% × {recov:.0f}%"
        st.markdown(
            f'<div class="note">'
            f'Throughput: <b>{tpa/1e6:.3f} Mt/yr</b> ({tpd:,.0f} tpd @ {avail:.0f}% avail)<br>'
            f'→ {formula}'
            f' = <b>{ann_pu:,.0f} {pul}/yr</b>'
            f' ({ann:,.4f} {unit}/yr)</div>',
            unsafe_allow_html=True)

        annual_prod  = ann
        throughput_tpd = tpd; avail_pct = avail; grade_value = grade_val; recovery_pct = recov
        opex_pu_val  = 0.

    else:
        # Study summary mode
        _md = {
            "Gold":(63847.,"%.2f",100.),"Silver":(500000.,"%.0f",10000.),
            "Copper":(23e6,"%.0f",1e5),"Iron Ore":(5e6,"%.0f",1e5),
            "Lithium":(10000.,"%.2f",10.),"Nickel":(15000.,"%.2f",10.),
            "Zinc":(50000.,"%.1f",100.),"Coal":(2e6,"%.0f",10000.),
            "Other":(10000.,"%.2f",10.),
        }
        _def, _fmt, _step = _md[commodity]
        annual_prod = st.number_input(
            f"Annual production  ({unit}/yr)",
            0.000001, 1e12, float(st.session_state.get("annual_prod", _def)), _step, format=_fmt, key="annual_prod",
            help=f"LOM average annual METAL production in {unit}.\n"
                 f"Enter the final recovered/payable figure from the study — not ore throughput.")

        # Warn if looks like ore tonnes
        if unit in ("t","kt","Mt") and pu in ("$/oz","$/g") and annual_prod > 500:
            st.markdown(
                f'<div class="warn">⚠ {annual_prod:,.1f} {unit} looks like ore throughput. '
                f'For gold/silver, annual metal production in tonnes is typically a small number '
                f'(e.g. 2–5 t Au/yr). Use Ore-based mode if entering throughput.</div>',
                unsafe_allow_html=True)

        throughput_tpd=0.; avail_pct=91.; grade_value=float(d[2])
        recovery_pct=float(d[3]); tpa=0.; rec_t=0.; ann=annual_prod; ann_pu=annual_prod*uc
        opex_pu_val = 0.

    # ── 5. Pricing ────────────────────────────────────────────────────────────
    st.markdown("#### Pricing")
    c1, c2 = st.columns(2)
    price   = c1.number_input(f"Price  ({pu})", 0.01, 1e8, float(st.session_state.get("commodity_price", float(d[0]))), float(d[1]), key="commodity_price")
    payable = c2.number_input("Payable  (%)",   0.0, 100.0, float(st.session_state.get("payable", float(d[7]))), 0.5, key="payable",
        help="% of produced metal the smelter/offtaker pays for. Not the same as recovery.")

    # Revenue preview
    rev_prev = annual_prod * uc * price * payable / 100 / 1e6
    st.markdown(
        f'<div class="note">→ <b>{annual_prod:,.4f} {unit}/yr</b>'
        f' × {uc:.4g} → {annual_prod*uc:,.0f} {pul}/yr'
        f' × ${price:.2f}/{pul} × {payable:.0f}%'
        f' = <b>${rev_prev:.1f}M/yr</b></div>',
        unsafe_allow_html=True)

    # ── 6. By-product ─────────────────────────────────────────────────────────
    st.markdown("#### By-product  *(optional)*")
    if use_ore:
        c1, c2 = st.columns(2)
        byp_grade = c1.number_input("Grade  (g/t)", 0., 100000., float(st.session_state.get("byp_grade", 0.0)), 0.1, key="byp_grade",
            help="By-product grade in g/t (e.g. gold credits in a copper mine). Leave 0 if none.")
        byp_rec   = c2.number_input("Recovery  (%)", 0., 100., float(st.session_state.get("byp_rec", 75.0)), 1., key="byp_rec")
        c1, c2 = st.columns(2)
        byp_price = c1.number_input("Price  ($/oz)", 0., 1e6, float(st.session_state.get("byp_price", 1800.0)), 10., key="byp_price")
        byp_pay   = c2.number_input("Payable  (%)",  0., 100., float(st.session_state.get("byp_pay", 90.0)),  0.5, key="byp_pay")
        if byp_grade > 0 and tpa > 0:
            byp_val = tpa * byp_grade * byp_rec/100 / 31.1035 * byp_price * byp_pay/100
            st.markdown(f'<div class="note">→ <b>${byp_val/1e6:.2f}M/yr</b></div>',
                        unsafe_allow_html=True)
        byp_rev = 0.
    else:
        byp_rev   = st.number_input("Revenue  ($M/yr)", 0., 1e5, float(st.session_state.get("byp_rev", 0.0)), 0.1, key="byp_rev",
            help="Net by-product credit in $M/yr already net of smelter charges.")
        byp_grade = byp_rec = byp_price = byp_pay = 0.

    # ── 7. Capital ────────────────────────────────────────────────────────────
    st.markdown("#### Capital  (US$M)")
    c1, c2 = st.columns(2)
    init_capex = c1.number_input("Initial CAPEX",  0.1, 1e6, float(st.session_state.get("initial_capex", 115.0)), 1., key="initial_capex")
    sust_mode  = c2.selectbox("Sustaining CAPEX", ["Total LOM","Per year"], index=_idx(["Total LOM","Per year"], st.session_state.get("sust_mode", "Total LOM")), key="sust_mode")
    c1, c2 = st.columns(2)
    sust_capex = c1.number_input("Sustaining CAPEX", 0., 1e4, float(st.session_state.get("sust_capex_input", 22.0)), 1., key="sust_capex_input")
    closure    = c2.number_input("Closure / rehab",  0., 1e4,  float(st.session_state.get("closure", 5.0)), 0.5, key="closure")
    sust_pa    = sust_capex / mine_life if sust_mode == "Total LOM" else sust_capex

    # ── 8. Operating costs ────────────────────────────────────────────────────
    st.markdown("#### Operating costs")

    # Single cost unit selector works for both modes
    opex_unit_type = st.radio(
        "Enter OPEX as",
        [f"Per tonne ore processed  ($/t)  — from DFS cost table",
         f"Per unit produced  ({pu})  — C1 or site cost"],
        index=_idx([f"Per tonne ore processed  ($/t)  — from DFS cost table", f"Per unit produced  ({pu})  — C1 or site cost"], st.session_state.get("opex_unit_type", f"Per tonne ore processed  ($/t)  — from DFS cost table" if st.session_state.get("opex_t", 0) > 0 else f"Per unit produced  ({pu})  — C1 or site cost")), key="opex_unit_type",
        horizontal=True, label_visibility="collapsed",
        help=f"$/t processed: from the DFS 'Total Operating Costs ($/t proc.)' breakdown\n"
             f"{pu}: C1 cash cost or AISC from the study summary. "
             f"If using C1 (which includes royalties), set royalty rate to 0%.")

    use_opex_t = opex_unit_type.startswith("Per tonne")

    c1, c2 = st.columns(2)
    if use_opex_t:
        opex_t_val = c1.number_input("Total OPEX  ($/t processed)", 0., 5000., float(st.session_state.get("opex_t", float(d[4]))), 0.01, key="opex_t",
            format="%.2f",
            help="Total site cost per tonne of ore — mining + processing + G&A combined.\n"
                 "e.g. Eva Copper: $53.65/t  |  Zelica Gold: ~$37/t")
        opex_pu_val = 0.
        if ann > 0 and uc > 0 and use_ore:
            equiv = opex_t_val * tpa / (ann * uc)
            st.markdown(
                f'<div class="note">${opex_t_val:.2f}/t ore × {tpa/1e6:.3f} Mt/yr'
                f' ÷ {ann_pu:,.0f} {pul}/yr = <b>${equiv:.2f}/{pul}</b> (before royalties)</div>',
                unsafe_allow_html=True)
        elif not use_ore:
            st.markdown(
                f'<div class="warn">$/t processed requires ore-based mode — switch to {pul} cost above, '
                f'or switch to Ore-based input mode.</div>',
                unsafe_allow_html=True)
            opex_t_val = 0.
    else:
        opex_pu_val = c1.number_input(
            f"Operating cost  ({pu})", 0., 1e8, float(st.session_state.get("opex_pu", float(d[5]))), float(d[6]), key="opex_pu",
            format="%.2f",
            help=f"Site operating cost per {pul} of metal produced.\n"
                 f"Use pure site OPEX + set royalty rate below, OR\n"
                 f"enter C1 cash cost + set royalty to 0% (C1 already includes royalties).")
        opex_t_val = 0.
        if use_ore and ann > 0 and uc > 0:
            implied_t = opex_pu_val * ann * uc / tpa if tpa > 0 else 0
            st.markdown(
                f'<div class="note">${opex_pu_val:.2f}/{pul} × {ann_pu:,.0f} {pul}/yr'
                f' ÷ {tpa/1e6:.3f} Mt/yr = <b>${implied_t:.2f}/t ore</b></div>',
                unsafe_allow_html=True)

    opex_esc = c2.number_input("Escalation  (%/yr)", -10., 20., float(st.session_state.get("opex_esc", 0.0)), 0.5, key="opex_esc")

    # ── 9. Fiscal ─────────────────────────────────────────────────────────────
    st.markdown("#### Fiscal")
    c1, c2 = st.columns(2)
    royalty_rate  = c1.number_input("Royalty rate  (%)",          0., 30.,  float(st.session_state.get("royalty_rate", 3.0)),  0.25, key="royalty_rate")
    tax_rate      = c2.number_input("Tax rate  (%)",              0., 60., float(st.session_state.get("tax_rate", 27.5)),  0.5, key="tax_rate")
    discount_rate = st.number_input("Discount rate / WACC  (%)", 0.5, 50.,  float(st.session_state.get("discount_rate", 8.0)),  0.5, key="discount_rate")

    st.markdown("#### Depreciation / tax treatment")
    dep_mode = st.selectbox(
        "Sustaining CAPEX tax treatment",
        ["Immediately deductible  (expense in year incurred)",
         "Add to depreciation pool  (straight-line, same life)"],
        index=_idx(["Immediately deductible  (expense in year incurred)", "Add to depreciation pool  (straight-line, same life)"], st.session_state.get("dep_mode", "Immediately deductible  (expense in year incurred)")), key="dep_mode",
        help="Immediately deductible: sustaining capex reduces taxable income in the year spent.\n"
             "Depreciation pool: added to the depreciable asset base and written off over mine life.\n"
             "Most junior study models use immediately deductible. Check your jurisdiction.")

    st.markdown("---")
    run_btn = st.button("▶  Run DCF Model", type="primary", use_container_width=True)


# ── Params ─────────────────────────────────────────────────────────────────────
params = dict(
    project_name=name, commodity=commodity, unit=unit, price_unit=price_unit,
    unit_conv=uc, input_mode="ore" if use_ore else "summary",
    mine_life=mine_life, construction_years=const_years, ramp_up=ramp_up,
    throughput_tpd=throughput_tpd, avail_pct=avail_pct,
    grade_value=grade_value, grade_system=grade_sys, recovery_pct=recovery_pct,
    opex_t=opex_t_val, opex_pu=opex_pu_val,
    byp_grade=byp_grade if use_ore else 0.,
    byp_rec=byp_rec   if use_ore else 0.,
    byp_price=byp_price if use_ore else 0.,
    byp_pay=byp_pay   if use_ore else 0.,
    annual_prod=annual_prod, byp_rev=byp_rev if not use_ore else 0.,
    commodity_price=price, payable=payable,
    price_esc=0., opex_esc=opex_esc,
    initial_capex=init_capex, sust_capex=sust_pa, closure=closure,
    royalty_rate=royalty_rate, tax_rate=tax_rate, discount_rate=discount_rate,
    dep_mode=dep_mode,
)


project_inputs = {
    "project_name": st.session_state.get("project_name", name),
    "commodity": st.session_state.get("commodity", commodity),
    "unit": st.session_state.get("unit", unit),
    "price_unit": st.session_state.get("price_unit", price_unit),
    "input_mode_label": st.session_state.get("input_mode_label", input_mode),
    "mine_life": st.session_state.get("mine_life", mine_life),
    "construction_years": st.session_state.get("construction_years", const_years),
    "ramp_up": st.session_state.get("ramp_up", ramp_up),
    "ore_input_type": st.session_state.get("ore_input_type", "Total LOM  (e.g. 1.2 Mt resource)"),
    "total_ore_mt": st.session_state.get("total_ore_mt", 1.2),
    "annual_mt": st.session_state.get("annual_mt", 0.6),
    "throughput_tpd": st.session_state.get("throughput_tpd", throughput_tpd),
    "avail_pct": st.session_state.get("avail_pct", avail_pct),
    "grade_value": st.session_state.get("grade_value", grade_value),
    "recovery_pct": st.session_state.get("recovery_pct", recovery_pct),
    "annual_prod": st.session_state.get("annual_prod", annual_prod),
    "commodity_price": st.session_state.get("commodity_price", price),
    "payable": st.session_state.get("payable", payable),
    "byp_grade": st.session_state.get("byp_grade", byp_grade if use_ore else 0.0),
    "byp_rec": st.session_state.get("byp_rec", byp_rec if use_ore else 0.0),
    "byp_price": st.session_state.get("byp_price", byp_price if use_ore else 0.0),
    "byp_pay": st.session_state.get("byp_pay", byp_pay if use_ore else 0.0),
    "byp_rev": st.session_state.get("byp_rev", byp_rev if not use_ore else 0.0),
    "sust_mode": st.session_state.get("sust_mode", sust_mode),
    "initial_capex": st.session_state.get("initial_capex", init_capex),
    "sust_capex_input": st.session_state.get("sust_capex_input", sust_capex),
    "closure": st.session_state.get("closure", closure),
    "opex_unit_type": st.session_state.get("opex_unit_type", opex_unit_type),
    "opex_t": st.session_state.get("opex_t", opex_t_val),
    "opex_pu": st.session_state.get("opex_pu", opex_pu_val),
    "opex_esc": st.session_state.get("opex_esc", opex_esc),
    "royalty_rate": st.session_state.get("royalty_rate", royalty_rate),
    "tax_rate": st.session_state.get("tax_rate", tax_rate),
    "discount_rate": st.session_state.get("discount_rate", discount_rate),
    "dep_mode": st.session_state.get("dep_mode", dep_mode),
}

project_download_payload = {
    "project_name": project_inputs.get("project_name", name),
    "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "model_file": "mining_dcf_app.py",
    "inputs": project_inputs,
}

# Apply dep_mode: if "pool", add sust_capex to depreciation base
# We handle this by adjusting the params before passing to build_schedule
if dep_mode.startswith("Add"):
    # Combine into one depreciable pool — engine uses initial_capex for dep calc
    # so we inflate initial_capex by total sust for depreciation purposes only
    params["dep_base"] = init_capex * 1e6 + sust_pa * mine_life * 1e6
    params["sust_for_dep"] = True
else:
    params["dep_base"] = init_capex * 1e6
    params["sust_for_dep"] = False


# ── Patch build_schedule to respect dep_mode ──────────────────────────────────
def build_schedule_v2(p):
    """Wrapper that adjusts depreciation base before calling engine."""
    p2 = deepcopy(p)
    # dep_ic used inside engine: override via dep_base
    return build_schedule(p2)

# Re-patch the engine depreciation to use dep_base
_orig_build = build_schedule
def build_schedule(p):
    import pandas as _pd
    import numpy as _np
    ml  = int(p["mine_life"]); cy = int(p.get("construction_years",0))
    dr  = float(p["discount_rate"])/100; tr = float(p["tax_rate"])/100
    rr  = float(p["royalty_rate"])/100
    pe  = float(p.get("price_esc",0))/100; oe = float(p.get("opex_esc",0))/100
    ramp= float(p.get("ramp_up",100))/100; pay = float(p.get("payable",100))/100
    ic  = float(p["initial_capex"])*1e6
    sc  = float(p.get("sust_capex",0))*1e6
    cl  = float(p.get("closure",0))*1e6
    uc2 = float(p.get("unit_conv",1.))
    byp_pa = float(p.get("byp_rev",0))*1e6

    mode = p.get("input_mode","summary")
    if mode == "ore":
        tpa = float(p["throughput_tpd"])*365*float(p.get("avail_pct",91))/100
        cont= grade_to_contained_t(tpa,float(p["grade_value"]),p["grade_system"])
        rec_t=cont*float(p["recovery_pct"])/100
        ann_base=rec_t/PROD_TO_T.get(p.get("unit","t"),1.)
        ann_pu=ann_base*uc2
        opex_tpa=float(p.get("opex_t",0))
        opex_pu_d=float(p.get("opex_pu",0))
        if opex_tpa>0:
            opex_base=(opex_tpa*tpa/ann_pu) if ann_pu>0 else 0.
        else:
            opex_base=opex_pu_d
        if float(p.get("byp_grade",0))>0:
            byp_oz=tpa*float(p["byp_grade"])*float(p.get("byp_rec",75))/100/31.1035
            byp_pa=byp_oz*float(p.get("byp_price",0))*float(p.get("byp_pay",90))/100
    else:
        ann_base=float(p["annual_prod"]); opex_base=float(p["opex_pu"]); tpa=0.

    # Depreciation: use dep_base if provided
    dep_base = float(p.get("dep_base", ic))
    dep_ic   = dep_base / ml if ml > 0 else 0.
    sust_deductible = not p.get("sust_for_dep", False)  # True = expense immediately

    records, loss, wc = [], 0., 0.
    for yr in range(1, ml+cy+1):
        op = yr - cy
        capex_yr=(ic/cy) if (cy>0 and op<=0) else (ic if yr==1 and cy==0 else 0.)
        sust_yr = sc if op>0 else 0.
        cl_yr   = cl if op==ml else 0.
        if   op<=0: prod=0.
        elif op==1: prod=ann_base*ramp
        else:       prod=ann_base
        esc   =max(op-1,0)
        price =float(p["commodity_price"])*(1+pe)**esc if op>0 else 0.
        opex_u=opex_base*(1+oe)**esc                   if op>0 else 0.
        gross =prod*uc2*price*pay
        total =gross+(byp_pa if op>0 else 0.)
        royal =gross*rr
        opex_tot=prod*uc2*opex_u
        ebitda=total-opex_tot-royal
        dep_yr=dep_ic if op>0 else 0.
        # Taxable income depends on treatment of sustaining capex
        if sust_deductible:
            taxable = (ebitda - dep_yr - sust_yr) - loss
        else:
            taxable = (ebitda - dep_yr) - loss
        tax_yr=max(taxable*tr,0.)
        if taxable<0: loss=abs(taxable)
        else: loss=max(loss-(ebitda-dep_yr-(sust_yr if sust_deductible else 0.)),0.)
        wc_yr=0.
        if op==1:    wc_yr=-(opex_tot*0.25); wc=-wc_yr
        elif op==ml: wc_yr=wc
        fcf=ebitda-tax_yr-capex_yr-sust_yr-cl_yr+wc_yr
        df_=1/(1+dr)**((op-0.5) if op>0 else (yr-0.5))
        records.append({"Year":f"Yr {yr}","Op.Yr":op,"Production":prod,
            "Price":price,"Revenue":total,"OPEX":opex_tot,"Royalties":royal,
            "EBITDA":ebitda,"Depreciation":dep_yr,"Tax":tax_yr,
            "Init.Capex":capex_yr,"Sust.Capex":sust_yr,
            "FCF":fcf,"DF":df_,"PV":fcf*df_})
    df=_pd.DataFrame(records)
    df["CumFCF"]=df["FCF"].cumsum(); df["CumPV"]=df["PV"].cumsum()
    return df, tpa, ann_base, opex_base


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header">
  <h1>⛏ Mining Project DCF Valuation</h1>
  <p>Select an input mode, fill in the sidebar, click Run</p>
</div>""", unsafe_allow_html=True)

if not run_btn and "res" not in st.session_state:
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px">
      <div style="background:white;border-radius:8px;padding:20px;border-top:3px solid #0f3460;box-shadow:0 2px 6px rgba(0,0,0,.06)">
        <b style="color:#0f3460">Study summary mode</b>
        <p style="font-size:.85em;color:#555;margin:6px 0 0">You have the final numbers from an ASX
        announcement or investor presentation — annual production, C1/OPEX, capex.
        Enter them directly. No grade or throughput needed.</p>
      </div>
      <div style="background:white;border-radius:8px;padding:20px;border-top:3px solid #c9a84c;box-shadow:0 2px 6px rgba(0,0,0,.06)">
        <b style="color:#0f3460">Ore-based mode</b>
        <p style="font-size:.85em;color:#555;margin:6px 0 0">You have the technical parameters from
        a feasibility study — throughput, head grade, recovery, $/t processed costs.
        The model calculates production and cost per unit automatically.</p>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if run_btn:
    errs = []
    if not name.strip():     errs.append("Project name required.")
    if mine_life < 1:        errs.append("Mine life must be ≥ 1 year.")
    if price <= 0:           errs.append("Commodity price must be > 0.")
    if init_capex <= 0:      errs.append("Initial CAPEX must be > 0.")
    if use_ore:
        if tpd <= 0:         errs.append("Throughput must be > 0.")
        if grade_val <= 0:   errs.append("Head grade must be > 0.")
        if not (0 < recov <= 100): errs.append("Recovery must be 1–100%.")
    else:
        if annual_prod <= 0: errs.append("Annual production must be > 0.")
    if errs:
        st.markdown('<div class="err"><b>Please fix:</b><br>'+"<br>".join(f"• {e}" for e in errs)+"</div>",
                    unsafe_allow_html=True)
        st.stop()
    with st.spinner("Running…"):
        df, tpa_r, ann_r, opex_r = build_schedule(params)
        k = kpis(df, params)
    st.session_state["res"] = (df, k, params, tpa_r, ann_r, opex_r)
else:
    df, k, params, tpa_r, ann_r, opex_r = st.session_state["res"]

ul  = params["unit"]
pu2 = params.get("price_unit", f"$/{ul}")
pul2= pu2.replace("$/","")
uc2 = float(params.get("unit_conv",1.))


tab1, tab2, tab3, tab4 = st.tabs(["📊 Results","🌪 Sensitivity","🎲 Monte Carlo","📥 Export"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    # Row 1: headline valuation metrics
    kd1 = [
        ("NPV (post-tax)",    f"${k['npv']:.1f}M",       "pos" if k["npv"]>=0 else "neg"),
        ("NPV (pre-tax)",     f"${k['npv_pretax']:.1f}M", "pos" if k.get("npv_pretax",0)>=0 else "neg"),
        ("IRR",               f"{k['irr']:.1f}%"          if not np.isnan(k["irr"]) else "N/A","neu"),
        ("Payback",           f"{k['pb']:.1f} yrs"        if not np.isnan(k["pb"])  else "N/A","neu"),
        ("NPV / CAPEX",       f"{k['mult']:.2f}x"         if not np.isnan(k["mult"]) else "N/A","neu"),
    ]
    # Row 2: operating metrics
    kd2 = [
        (f"AISC  ({pu2})",    f"${k['aisc']:.2f}",    "neu"),
        (f"C1 cost  ({pu2})", f"${k['c1']:.2f}",      "neu"),
        ("EBITDA margin",      f"{k['eb']:.1f}%",      "pos" if k["eb"]>0 else "neg"),
        ("EBITDA",             f"${k['tebitda']:.0f}M","neu"),
        ("Total FCF",          f"${k['tfcf']:.0f}M",   "pos" if k.get('tfcf',0)>=0 else "neg"),
    ]
    cols = st.columns(5)
    for col,(lbl,val,cls) in zip(cols,kd1):
        with col:
            st.markdown(f'<div class="kcard"><div class="klbl">{lbl}</div>'
                        f'<div class="kval {cls}">{val}</div></div>',unsafe_allow_html=True)
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for col,(lbl,val,cls) in zip(cols,kd2):
        with col:
            st.markdown(f'<div class="kcard"><div class="klbl">{lbl}</div>'
                        f'<div class="kval {cls}" style="font-size:1.35em">{val}</div></div>',unsafe_allow_html=True)

    # Technical summary for ore mode
    if params.get("input_mode")=="ore":
        opex_pu2 = params["opex_t"]*tpa_r/(ann_r*uc2) if ann_r*uc2>0 else 0
        gdisp = f"{params['grade_value']:.3f} {'g/t' if params['grade_system']=='g/t' else '%'}"
        st.markdown(
            f'<div class="note" style="margin-top:12px"><b>Technical:</b> '
            f'{tpa_r/1e6:.3f} Mt/yr ore · {gdisp} · {params["recovery_pct"]:.1f}% recovery'
            f' → <b>{ann_r:,.2f} {ul}/yr</b> ({ann_r*uc2:,.0f} {pul2}/yr)'
            f' · ${params["opex_t"]:.2f}/t = <b>${opex_pu2:.3f}/{pul2}</b> (before royalties)'
            f' · C1: <b>${k["c1"]:.3f}/{pul2}</b></div>',
            unsafe_allow_html=True)

    dep_note = ("Sustaining capex immediately deductible for tax"
                if not params.get("sust_for_dep")
                else "Sustaining capex added to depreciation pool")

    st.markdown('<div class="shdr">Cash flow analysis</div>', unsafe_allow_html=True)
    op = df[df["Op.Yr"]>0].copy(); x = np.arange(len(op))
    C  = {"b":"#2471a3","r":"#c0392b","g":"#1e8449","n":"#0f3460",
          "o":"#d35400","p":"#7d3c98","gold":"#c9a84c","gr":"#7f8c8d"}
    fig,axes = plt.subplots(1,3,figsize=(15,4.2)); fig.patch.set_facecolor("#f9fafb")

    ax=axes[0]; w=0.27
    ax.bar(x-w,op["Revenue"].values/1e6, w,color=C["b"],alpha=.85,label="Revenue")
    ax.bar(x,  op["OPEX"].values/1e6,    w,color=C["r"],alpha=.85,label="OPEX")
    ax.bar(x+w,op["EBITDA"].values/1e6,  w,color=C["g"],alpha=.85,label="EBITDA")
    ax.set_xticks(x); ax.set_xticklabels(op["Op.Yr"].astype(str),fontsize=7,rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
    ax.set_title("Revenue / OPEX / EBITDA",fontsize=9,fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y",alpha=.25,ls="--"); ax.set_facecolor("#f9fafb")

    ax=axes[1]
    bc=[C["n"] if v>=0 else C["r"] for v in op["FCF"]]
    ax.bar(x,op["FCF"].values/1e6,color=bc,alpha=.8,label="FCF")
    ax.axhline(0,color="black",lw=.6)
    ax2=ax.twinx()
    ax2.plot(np.arange(len(df))-(len(df)-len(op)),df["CumPV"].values/1e6,
             color=C["gold"],lw=2,label="Cum. NPV")
    ax2.axhline(0,color=C["gold"],lw=.5,ls=":")
    ax.set_xticks(x); ax.set_xticklabels(op["Op.Yr"].astype(str),fontsize=7,rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
    ax2.set_ylabel("Cum. NPV",fontsize=7,color=C["gold"])
    ax.set_title("FCF & Cumulative NPV",fontsize=9,fontweight="bold")
    ax.grid(axis="y",alpha=.25,ls="--"); ax.set_facecolor("#f9fafb")
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=7)

    ax=axes[2]
    rev=k["tr"]; opx=k["to"]
    roy=op["Royalties"].sum()/1e6; sust=op["Sust.Capex"].sum()/1e6
    tax=k["tt"]; capx=float(params["initial_capex"])
    net=rev-opx-roy-sust-tax-capx
    items=[("Revenue",rev,C["b"]),("OPEX",-opx,C["r"]),("Royalties",-roy,C["o"]),
           ("Sust. Cap.",-sust,C["gr"]),("Tax",-tax,C["p"]),
           ("Init. Cap.",-capx,C["r"]),("Net FCF",net,C["g"] if net>=0 else C["r"])]
    run=0
    for lbl,val,col in items:
        if lbl=="Net FCF": ax.bar(lbl,abs(val),bottom=0 if val>=0 else val,color=col,alpha=.9)
        else:
            bot=run if val>=0 else run+val
            ax.bar(lbl,abs(val),bottom=bot,color=col,alpha=.85); run+=val
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
    ax.set_title("Lifetime Waterfall",fontsize=9,fontweight="bold")
    ax.tick_params(axis="x",rotation=30,labelsize=7)
    ax.grid(axis="y",alpha=.25,ls="--"); ax.set_facecolor("#f9fafb")
    plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown('<div class="shdr">Annual cash flow schedule</div>', unsafe_allow_html=True)
    dfd=df[["Year","Op.Yr","Production","Price","Revenue","OPEX","EBITDA",
            "Depreciation","Tax","FCF","PV","CumPV"]].copy()
    dfd["Production"]=df["Production"].apply(lambda v:f"{v:,.4f} {ul}")
    dfd["Price"]=df["Price"].apply(lambda v:f"${v:,.2f}" if v>0 else "—")
    for c in ["Revenue","OPEX","EBITDA","Depreciation","Tax","FCF","PV","CumPV"]:
        dfd[c]=df[c].apply(lambda v:f"${v/1e6:,.2f}M")
    dfd.columns=["Year","Op.Yr","Production",f"Price ({pu2})","Revenue","OPEX","EBITDA",
                 "Depreciation","Tax","Free Cash Flow","PV of FCF","Cum. NPV"]
    st.dataframe(dfd,use_container_width=True,hide_index=True)
    st.markdown(
        f'<div class="foot">Mid-year discounting · Initial capex straight-line depreciation · '
        f'{dep_note} · Carry-forward losses · Working capital 25% of Year-1 OPEX</div>',
        unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    SENS = {"Commodity Price":"commodity_price","Initial CAPEX ($M)":"initial_capex",
            "Discount Rate (%)":"discount_rate","Royalty Rate (%)":"royalty_rate"}
    if params.get("input_mode")=="ore":
        SENS["Head Grade"]        = "grade_value"
        SENS["Recovery (%)"]      = "recovery_pct"
        SENS["OPEX ($/t proc.)"]  = "opex_t"
    else:
        SENS["Annual Production"] = "annual_prod"
        SENS["Operating Cost"]    = "opex_pu"

    base_npv = k["npv"]

    def _s(p2, key, f):
        p2[key] = float(params.get(key,0)) * f
    def _run(p2):
        try: return kpis(build_schedule(p2)[0],p2)["npv"]
        except: return base_npv

    # Tornado
    st.markdown('<div class="shdr">Tornado — ±20%</div>', unsafe_allow_html=True)
    rows=[]
    for lbl,key in SENS.items():
        r={}
        for sign,tag in [(-1,"dn"),(1,"up")]:
            p2=deepcopy(params); _s(p2,key,1+sign*.20)
            r[tag]=_run(p2)
        lo,hi=min(r["up"],r["dn"]),max(r["up"],r["dn"])
        rows.append({"lbl":lbl,"lo":lo,"hi":hi,"rng":hi-lo})
    rows.sort(key=lambda r:r["rng"])
    fig,ax=plt.subplots(figsize=(9,max(3.5,len(rows)*.7)))
    fig.patch.set_facecolor("#f9fafb"); ax.set_facecolor("#f9fafb")
    for i,r in enumerate(rows):
        ax.barh(i,r["hi"]-base_npv,left=base_npv,color="#1e8449",alpha=.8,height=.52)
        ax.barh(i,r["lo"]-base_npv,left=base_npv,color="#c0392b",alpha=.8,height=.52)
        pad=max(r["rng"]*.015,.5)
        ax.text(r["hi"]+pad,i,f"${r['hi']:.0f}M",va="center",fontsize=7.5,color="#1e8449")
        ax.text(r["lo"]-pad,i,f"${r['lo']:.0f}M",va="center",ha="right",fontsize=7.5,color="#c0392b")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["lbl"] for r in rows],fontsize=9)
    ax.axvline(base_npv,color="black",lw=1.8,ls="--")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
    ax.set_title(f"Tornado — ±20%  (base NPV ${base_npv:.0f}M)",fontsize=10,fontweight="bold")
    ax.grid(axis="x",alpha=.2,ls="--"); plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close("all")

    # Spider
    st.markdown('<div class="shdr">Spider diagram — ±30%</div>', unsafe_allow_html=True)
    pcts=np.linspace(-.30,.30,9)
    fig,ax=plt.subplots(figsize=(9,5))
    fig.patch.set_facecolor("#f9fafb"); ax.set_facecolor("#f9fafb")
    cmap=plt.get_cmap("tab10")
    for i,(lbl,key) in enumerate(SENS.items()):
        line=[]
        for pct in pcts:
            p2=deepcopy(params); _s(p2,key,1+pct); line.append(_run(p2))
        ax.plot(pcts*100,line,marker="o",markersize=3.5,lw=1.8,label=lbl,color=cmap(i))
    ax.axvline(0,color="grey",lw=.7,ls=":")
    ax.axhline(base_npv,color="black",lw=.8,ls="--",alpha=.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
    ax.set_xlabel("% change",fontsize=9); ax.set_ylabel("NPV ($M)",fontsize=9)
    ax.set_title(f"Spider — ±30%  (base NPV ${base_npv:.0f}M)",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=.2,ls="--"); plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close("all")

    # Scenario matrix
    st.markdown('<div class="shdr">Price × OPEX scenario matrix</div>', unsafe_allow_html=True)
    ok = "opex_t" if params.get("input_mode")=="ore" else "opex_pu"
    st.caption(f"NPV at ±30% combinations of price and operating cost. Dashed = base case.")
    if st.button("Generate matrix",type="primary"):
        pps=np.linspace(-.30,.30,7); ops=np.linspace(-.30,.30,7)
        bp=float(params["commodity_price"]); bo=float(params.get(ok,0))
        mat=np.zeros((7,7))
        for i,pp in enumerate(pps):
            for j,op_ in enumerate(ops):
                p2=deepcopy(params); p2["commodity_price"]=bp*(1+pp); p2[ok]=bo*(1+op_)
                try: mat[i,j]=kpis(build_schedule(p2)[0],p2)["npv"]
                except: mat[i,j]=float("nan")
        pl=[f"{p*100:+.0f}%" for p in pps]; ol=[f"{o*100:+.0f}%" for o in ops]
        vn,vx=np.nanmin(mat),np.nanmax(mat)
        norm=mcolors.TwoSlopeNorm(vmin=vn,vcenter=max(0,vn+.001),vmax=max(vx,vn+.01))
        fig,ax=plt.subplots(figsize=(8,6)); fig.patch.set_facecolor("#f9fafb")
        im=ax.imshow(mat,cmap="RdYlGn",norm=norm,aspect="auto")
        plt.colorbar(im,ax=ax,label="NPV ($M)")
        for i in range(7):
            for j in range(7):
                ax.text(j,i,f"${mat[i,j]:.0f}M",ha="center",va="center",fontsize=7.5,
                        color="white" if abs(mat[i,j])>abs(np.nanmean(mat))*1.2 else "#1a1a1a")
        ax.set_xticks(range(7)); ax.set_xticklabels(ol,fontsize=8)
        ax.set_yticks(range(7)); ax.set_yticklabels(pl,fontsize=8)
        ax.set_xlabel("Operating cost change",fontsize=9)
        ax.set_ylabel("Price change",fontsize=9)
        ax.set_title("NPV ($M)",fontsize=9,fontweight="bold")
        ax.add_patch(plt.Rectangle((2.5,2.5),1,1,fill=False,edgecolor="black",lw=2.5,ls="--"))
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close("all")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<div class="shdr">Monte Carlo — 10,000 trials</div>', unsafe_allow_html=True)
    st.caption("Price: lognormal · Grade/production: lognormal · OPEX & CAPEX: triangular")
    c1,c2,c3=st.columns(3)
    pv  = c1.slider("Price volatility (%)",5,40,12)/100
    gv  = c1.slider("Grade / production vol. (%)",5,40,15)/100
    olo = c2.slider("OPEX best case (%)",0,30,10)/100
    ohi = c2.slider("OPEX worst case (%)",0,40,20)/100
    clo = c3.slider("CAPEX best case (%)",0,30,10)/100
    chi = c3.slider("CAPEX worst case (%)",0,40,20)/100

    if st.button("🎲 Run Monte Carlo",type="primary"):
        rng=np.random.default_rng(42)
        bp=float(params["commodity_price"]); bc=float(params["initial_capex"])
        ok2="opex_t" if params.get("input_mode")=="ore" else "opex_pu"
        bo=float(params.get(ok2,0)); npvs=[]
        for _ in range(10000):
            p2=deepcopy(params)
            p2["commodity_price"]=np.exp(rng.normal(np.log(bp)-.5*pv**2,pv))
            p2[ok2]=bo*rng.triangular(1-olo,1.,1+ohi)
            p2["initial_capex"]=bc*rng.triangular(1-clo,1.,1+chi)
            if params.get("input_mode")=="ore":
                bg=float(params.get("grade_value",0)); br=float(params.get("recovery_pct",0))
                p2["grade_value"]=max(np.exp(rng.normal(np.log(max(bg,1e-9))-.5*gv**2,gv)),bg*.1)
                p2["recovery_pct"]=min(max(br*rng.normal(1.,.08),10),99)
            else:
                bprod=float(params.get("annual_prod",0))
                p2["annual_prod"]=max(np.exp(rng.normal(np.log(max(bprod,1e-9))-.5*gv**2,gv)),bprod*.1)
            try: npvs.append(kpis(build_schedule(p2)[0],p2)["npv"])
            except: pass
        npvs=np.array(npvs)
        pcts_mc=np.percentile(npvs,[10,50,90])
        st.session_state["mc"]=(npvs,pcts_mc,(npvs>0).mean()*100)

    if "mc" in st.session_state:
        npvs,pcts_mc,pp=st.session_state["mc"]
        cols=st.columns(6)
        for col,(lbl,val) in zip(cols,[
            ("Mean NPV",f"${npvs.mean():.0f}M"),("Median P50",f"${np.median(npvs):.0f}M"),
            ("Std dev",f"${npvs.std():.0f}M"),("P(NPV>0)",f"{pp:.1f}%"),
            ("P10",f"${pcts_mc[0]:.0f}M"),("P90",f"${pcts_mc[2]:.0f}M")]):
            with col:
                st.markdown(f'<div class="kcard"><div class="klbl">{lbl}</div>'
                            f'<div class="kval neu" style="font-size:1.3em">{val}</div></div>',
                            unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,axes=plt.subplots(1,2,figsize=(13,4.2)); fig.patch.set_facecolor("#f9fafb")
        ax=axes[0]; ax.set_facecolor("#f9fafb")
        ax.hist(npvs,bins=80,color="#0f3460",alpha=.75,edgecolor="white",lw=.3)
        for pv2,lbl,col in [(pcts_mc[0],"P10","#c0392b"),(pcts_mc[1],"P50","#d35400"),(pcts_mc[2],"P90","#1e8449")]:
            ax.axvline(pv2,color=col,lw=1.8,ls="--")
            ax.text(pv2,ax.get_ylim()[1]*.88,f" {lbl}\n ${pv2:.0f}M",color=col,fontsize=8,va="top")
        ax.axvline(k["npv"],color="black",lw=2)
        ax.text(k["npv"],ax.get_ylim()[1]*.97,f" Base\n ${k['npv']:.0f}M",fontsize=8,va="top")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
        ax.set_title(f"NPV Distribution  —  P(NPV>0) = {pp:.1f}%",fontsize=9,fontweight="bold")
        ax.grid(axis="y",alpha=.2,ls="--")
        ax=axes[1]; ax.set_facecolor("#f9fafb")
        sn=np.sort(npvs); cdf=np.arange(1,len(sn)+1)/len(sn)*100
        ax.plot(sn,cdf,color="#0f3460",lw=2)
        ax.fill_betweenx(cdf,sn,0,where=(sn<0),alpha=.12,color="#c0392b")
        ax.axvline(0,color="#c0392b",lw=1.2,ls="--",alpha=.7)
        ax.axhline(50,color="#7f8c8d",lw=.7,ls=":")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"${v:.0f}M"))
        ax.set_title("S-Curve",fontsize=9,fontweight="bold")
        ax.set_ylabel("Cumulative probability (%)",fontsize=8)
        ax.grid(alpha=.2,ls="--")
        stats=(f"Mean:   ${npvs.mean():.0f}M\nMedian: ${np.median(npvs):.0f}M\n"
               f"Std:    ${npvs.std():.0f}M\nP10:    ${pcts_mc[0]:.0f}M\nP90:    ${pcts_mc[2]:.0f}M")
        ax.text(.97,.05,stats,transform=ax.transAxes,fontsize=8,va="bottom",ha="right",
                bbox=dict(boxstyle="round",facecolor="wheat",alpha=.7))
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close("all")
    else:
        st.info("Set the sliders above and click **Run Monte Carlo**.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown('<div class="shdr">Export</div>', unsafe_allow_html=True)
    st.download_button("💾 Download project (.json)",
                       data=json.dumps(project_download_payload, indent=2).encode("utf-8"),
                       file_name=f"{project_inputs.get('project_name','project').replace(' ','_')}_inputs.json",
                       mime="application/json",
                       use_container_width=True)
    c1,c2=st.columns(2)
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as wr:
        pd.DataFrame({
            "Metric":["Project","Commodity","Production unit","Price unit",
                      "Input mode","Mine life","Head grade","Recovery",
                      "NPV ($M)","IRR (%)","Payback (yrs)",
                      f"AISC ({pu2})",f"C1 ({pu2})",
                      "EBITDA margin (%)","NPV/CAPEX (x)",
                      f"Total production ({ul})","Total revenue ($M)",
                      "Total OPEX ($M)","Total tax ($M)","Dep. treatment"],
            "Value":[params["project_name"],params["commodity"],ul,pu2,
                     params.get("input_mode",""),
                     params["mine_life"],
                     f"{params['grade_value']:.3f} {'g/t' if params.get('grade_system')=='g/t' else '%'}",
                     f"{params['recovery_pct']:.1f}%",
                     round(k["npv"],2),
                     round(k["irr"],2) if not np.isnan(k["irr"]) else "N/A",
                     round(k["pb"],2)  if not np.isnan(k["pb"])  else "N/A",
                     round(k["aisc"],4),round(k["c1"],4),
                     round(k["eb"],1),
                     round(k["mult"],2) if not np.isnan(k["mult"]) else "N/A",
                     round(k["tp"],4),round(k["tr"],2),round(k["to"],2),round(k["tt"],2),
                     params.get("dep_mode","")]
        }).to_excel(wr,sheet_name="Summary",index=False)
        pd.DataFrame({"Parameter":list(params.keys()),"Value":list(params.values())
                     }).to_excel(wr,sheet_name="Inputs",index=False)
        df.to_excel(wr,sheet_name="Cash Flow",index=False)
        for sn in wr.sheets:
            ws=wr.sheets[sn]
            for col in ws.columns:
                w=max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width=min(w+3,30)
    buf.seek(0)
    with c1:
        st.markdown("**📊 Excel workbook**")
        st.download_button("📥 Download .xlsx",data=buf,
            file_name=f"{params['project_name'].replace(' ','_')}_DCF.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with c2:
        st.markdown("**📄 CSV — cash flows**")
        st.download_button("📥 Download .csv",data=df.to_csv(index=False).encode(),
            file_name=f"{params['project_name'].replace(' ','_')}_cashflows.csv",
            mime="text/csv",use_container_width=True)

    st.markdown("---")
    st.table(pd.DataFrame({
        "Metric":["Project","Commodity","Production unit","Price unit",
                  "Grade","Recovery","Mine life",
                  "NPV","IRR","Payback",
                  f"AISC ({pu2})",f"C1 ({pu2})",
                  "EBITDA margin","NPV / CAPEX",
                  f"LOM production ({ul})","Total revenue"],
        "Value":[params["project_name"],params["commodity"],ul,pu2,
                 f"{params['grade_value']:.3f} {'g/t' if params.get('grade_system')=='g/t' else '%'}",
                 f"{params['recovery_pct']:.1f}%",
                 f"{params['mine_life']} yrs",
                 f"${k['npv']:.2f}M",
                 f"{k['irr']:.2f}%" if not np.isnan(k["irr"]) else "N/A",
                 f"{k['pb']:.2f} yrs" if not np.isnan(k["pb"]) else "N/A",
                 f"${k['aisc']:.4f}",f"${k['c1']:.4f}",
                 f"{k['eb']:.1f}%",
                 f"{k['mult']:.2f}x" if not np.isnan(k["mult"]) else "N/A",
                 f"{k['tp']:,.4f}",f"${k['tr']:.1f}M"]}))

st.markdown("---")
st.markdown('<div style="text-align:center;color:#95a5a6;font-size:.72em;padding:6px 0">'
            'Mining Project DCF Valuation · For evaluation purposes only</div>',
            unsafe_allow_html=True)
