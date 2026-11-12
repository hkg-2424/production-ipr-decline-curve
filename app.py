"""
app.py
------
Streamlit Interactive Petroleum Engineering Calculator & Plotter:
- Inflow Performance Relationship (Composite Darcy-Vogel & Skin Sensitivity)
- Nodal Analysis (VLP Curves & Operating Point via Brent's Method)
- Arps Decline Curve Analysis (Exponential, Hyperbolic, Harmonic & EUR)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.arps_dca import fit_decline_curves
from src.ipr_skin import calculate_productivity_index, composite_darcy_vogel_rate, flow_efficiency
from src.nodal_operating_point import API_TUBING_SIZES, find_nodal_operating_point, tubing_intake_vlp_pressure

st.set_page_config(page_title="Petroleum Production & DCA Tool", layout="wide", page_icon="🛢️")

st.title("🛢️ Inflow Performance (IPR), Nodal Analysis & Arps DCA Tool")
st.caption("Standard Undergraduate Petroleum Engineering Formulations • Interactive Engineering Workbench")

tab_ipr, tab_nodal, tab_dca = st.tabs(["1. Inflow Performance (IPR)", "2. Nodal Analysis (VLP)", "3. Arps DCA & EUR"])

# ==================== TAB 1: IPR & SKIN FACTOR ====================
with tab_ipr:
    st.subheader("Composite Darcy-Vogel IPR & Skin Factor Sensitivity")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Reservoir & Fluid Parameters")
        pr = st.slider("Reservoir Pressure (Pr, psi)", 1500, 5000, 3800, step=50)
        pb = st.slider("Bubble Point Pressure (Pb, psi)", 1000, 4000, 2200, step=50)
        skin = st.slider("Mechanical Skin Factor (S)", -4.0, 10.0, 0.0, step=0.5)

        with st.expander("Rock & Fluid Properties", expanded=False):
            k = st.number_input("Permeability, k (mD)", 5.0, 500.0, 50.0, step=5.0)
            h = st.number_input("Net Thickness, h (ft)", 10.0, 300.0, 50.0, step=5.0)
            mu = st.number_input("Viscosity, μ (cP)", 0.2, 20.0, 1.2, step=0.1)
            bo = st.number_input("Volume Factor, Bo (rb/STB)", 1.0, 2.0, 1.25, step=0.05)
            re = st.number_input("Drainage Radius, re (ft)", 200.0, 5000.0, 1490.0, step=50.0)
            rw = st.number_input("Wellbore Radius, rw (ft)", 0.1, 1.0, 0.328, step=0.01)

    j_val = calculate_productivity_index(k, h, mu, bo, re, rw, skin=skin)
    fe_val = flow_efficiency(re, rw, skin=skin)
    pb_eff = min(pb, pr)
    qb = j_val * (pr - pb_eff) if pr > pb_eff else 0.0
    qv = (j_val * pb_eff) / 1.8 if pb_eff > 0 else 0.0
    aof = qb + qv

    with col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Productivity Index (J)", f"{j_val:.3f} STB/d/psi")
        m2.metric("Flow Efficiency (FE)", f"{fe_val:.2f}")
        m3.metric("Rate @ Pb (qb)", f"{qb:.0f} BOPD")
        m4.metric("Absolute Open Flow (AOF)", f"{aof:.0f} BOPD")

        pwf_grid = np.linspace(pr, 0, 100)
        q_user = composite_darcy_vogel_rate(pwf_grid, pr, pb, j_val)

        # Baseline S=0 and Stimulated S=-3 for direct comparison
        j_base = calculate_productivity_index(k, h, mu, bo, re, rw, skin=0.0)
        j_stim = calculate_productivity_index(k, h, mu, bo, re, rw, skin=-3.0)
        q_base = composite_darcy_vogel_rate(pwf_grid, pr, pb, j_base)
        q_stim = composite_darcy_vogel_rate(pwf_grid, pr, pb, j_stim)

        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        ax.plot(q_user, pwf_grid, label=f"Current Skin (S = {skin:+.1f})", color="#0275d8", lw=2.5)
        if abs(skin - 0.0) > 0.05:
            ax.plot(q_base, pwf_grid, label="Base Case (S = 0)", color="gray", lw=1.8, ls="--")
        if abs(skin - (-3.0)) > 0.05:
            ax.plot(q_stim, pwf_grid, label="Stimulated (S = -3)", color="#5cb85c", lw=1.8, ls="-.")
        ax.axhline(pb, color="crimson", ls=":", label=f"Bubble Point (Pb = {pb} psi)")
        ax.set_xlabel("Oil Rate, qo (STB/day)", fontweight="bold")
        ax.set_ylabel("Flowing Bottomhole Pressure, Pwf (psi)", fontweight="bold")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ==================== TAB 2: NODAL ANALYSIS ====================
with tab_nodal:
    st.subheader("Nodal Analysis: Reservoir IPR & Tubing Vertical Lift Performance (VLP)")
    ncol1, ncol2 = st.columns([1, 2])

    with ncol1:
        st.markdown("#### Wellbore & Lift Geometry")
        pwh = st.slider("Wellhead Pressure (Pwh, psi)", 50, 500, 150, step=25)
        tvd = st.slider("True Vertical Depth (TVD, ft)", 3000, 14000, 8000, step=250)
        gamma = st.slider("Fluid Specific Gravity (Water = 1.0)", 0.65, 1.10, 0.85, step=0.05)
        sel_tubings = st.multiselect("Active Tubing Sizes to Compare", list(API_TUBING_SIZES.keys()), default=list(API_TUBING_SIZES.keys()))

    with ncol2:
        q_nodal = np.linspace(1e-2, aof, 120)
        pwf_ipr_curve = np.array([float(composite_darcy_vogel_rate(p, pr, pb, j_val)) for p in np.linspace(pr, 0, 120)])

        fig2, ax2 = plt.subplots(figsize=(8, 4.2), dpi=150)
        # Inflow curve (re-evaluated cleanly)
        pwf_range = np.linspace(pr, 0, 150)
        q_inflow = composite_darcy_vogel_rate(pwf_range, pr, pb, j_val)
        ax2.plot(q_inflow, pwf_range, label=f"IPR (S={skin:+.1f})", color="black", lw=2.5)

        palette = ["#f0ad4e", "#5bc0de", "#6f42c1", "#d9534f"]
        nodal_table = []
        for idx, t_name in enumerate(sel_tubings):
            tid = API_TUBING_SIZES[t_name]
            vlp_pts = tubing_intake_vlp_pressure(q_inflow, pwh, tvd, tid, gamma, mu)
            c = palette[idx % len(palette)]
            ax2.plot(q_inflow, vlp_pts, label=f"VLP {t_name}", color=c, lw=1.8, ls="--")

            q_op, pwf_op, status = find_nodal_operating_point(pr, pb, j_val, pwh, tvd, tid, gamma, mu)
            if q_op is not None:
                ax2.scatter([q_op], [pwf_op], color=c, s=70, zorder=6, edgecolors="black")
                nodal_table.append({"Tubing": t_name, "Operating Rate (BOPD)": f"{q_op:.1f}", "Operating Pwf (psi)": f"{pwf_op:.1f}", "Status": status})
            else:
                nodal_table.append({"Tubing": t_name, "Operating Rate (BOPD)": "None", "Operating Pwf (psi)": "None", "Status": status})

        ax2.set_xlabel("Production Rate, q (STB/day)", fontweight="bold")
        ax2.set_ylabel("Pressure, Pwf (psi)", fontweight="bold")
        ax2.set_xlim(0, aof * 1.05)
        ax2.set_ylim(bottom=1500, top=pr * 1.08)
        ax2.legend(loc="lower left", frameon=True)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

        st.dataframe(pd.DataFrame(nodal_table))

# ==================== TAB 3: ARPS DECLINE CURVE ====================
with tab_dca:
    st.subheader("Arps Decline Curve Analysis & Estimated Ultimate Recovery (EUR)")
    dcol1, dcol2 = st.columns([1, 2])

    with dcol1:
        st.markdown("#### DCA Settings & Dataset Selection")
        dataset_choice = st.selectbox(
            "Select Field Dataset",
            [
                "Synthetic 36-Month Reservoir Base Case",
                "Equinor Volve Field 15/9-F-12 (Primary Decline 71-Mo)",
                "Equinor Volve Field 15/9-F-12 (Full 104-Mo History)",
                "Upload Custom CSV",
            ],
        )
        q_econ = st.slider("Economic Limit Rate (q_econ, BOPD)", 5.0, 100.0, 25.0, step=5.0)

        synth_path = "data/monthly_production_history.csv"
        volve_path = "data/real_field_benchmarks/real_volve_15_9_F12_monthly.csv"

        if dataset_choice == "Equinor Volve Field 15/9-F-12 (Primary Decline 71-Mo)":
            raw_v = pd.read_csv(volve_path)
            p_idx = raw_v["Oil_Rate_bopd"].idxmax()
            df_prod = raw_v.iloc[p_idx:p_idx + 71].copy().reset_index(drop=True)
            df_prod["Time_years"] = np.arange(len(df_prod)) / 12.0
            st.caption("Loaded Equinor Volve Well 15/9-F-12 Primary Waterflood Decline (71 months).")
        elif dataset_choice == "Equinor Volve Field 15/9-F-12 (Full 104-Mo History)":
            df_prod = pd.read_csv(volve_path)
            st.caption("Loaded Equinor Volve Well 15/9-F-12 Complete History (104 months).")
        elif dataset_choice == "Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload Custom CSV", type=["csv"])
            if uploaded_file is not None:
                df_prod = pd.read_csv(uploaded_file)
            else:
                df_prod = pd.read_csv(synth_path)
        else:
            df_prod = pd.read_csv(synth_path)
            st.caption(f"Loaded {len(df_prod)} synthetic production records.")

    fits = fit_decline_curves(df_prod["Time_years"].values, df_prod["Oil_Rate_bopd"].values, q_econ=q_econ)

    with dcol2:
        t_hist = df_prod["Time_years"].values
        q_hist = df_prod["Oil_Rate_bopd"].values

        max_t = min(30.0, max(f["t_econ_years"] for f in fits.values() if f["b"] < 1.0) * 1.15)
        t_plot = np.linspace(0.0833, max_t, 250)

        fig3, ax3 = plt.subplots(figsize=(8, 4.2), dpi=150)
        ax3.scatter(t_hist, q_hist, color="#0275d8", s=30, alpha=0.8, label="Historical Data", zorder=5)

        colors_map = {"Exponential (b=0)": "#5cb85c", "Hyperbolic (0<b<1)": "#d9534f", "Harmonic (b=1)": "#6f42c1"}
        dca_summary = []
        for name, m in fits.items():
            ax3.plot(t_plot, m["model_func"](t_plot), label=f"{name} (R²={m['r2']:.3f})", color=colors_map[name], lw=2.0)
            dca_summary.append({
                "Model": name,
                "qi (BOPD)": f"{m['qi']:.1f}",
                "Di (/yr)": f"{m['di']:.4f}",
                "b Exponent": f"{m['b']:.2f}",
                "R² Fit": f"{m['r2']:.4f}",
                "Time to Limit (yr)": f"{m['t_econ_years']:.2f}",
                "EUR (MMSTB)": f"{m['eur_stb']/1e6:.3f}",
            })

        ax3.axhline(q_econ, color="crimson", ls=":", label=f"Economic Limit ({q_econ:.0f} BOPD)")
        ax3.set_xlabel("Time (Years)", fontweight="bold")
        ax3.set_ylabel("Oil Rate (BOPD)", fontweight="bold")
        ax3.set_xlim(0, max_t)
        ax3.set_ylim(0, max(q_hist) * 1.15)
        ax3.legend(loc="upper right", frameon=True)
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

        st.dataframe(pd.DataFrame(dca_summary))
