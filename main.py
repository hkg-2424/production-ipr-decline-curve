"""
main.py
-------
Inflow Performance (IPR), Nodal Analysis & Arps Decline Curve Analysis CLI Tool.

Executes:
1. Composite Darcy-Vogel IPR with Skin Factor Sensitivity (S = +5, S = 0, S = -3).
2. Tubing Intake (VLP) curves & Nodal Operating Points for standard tubing sizes.
3. Arps Decline Curve fitting (Exponential, Hyperbolic, Harmonic) and EUR estimation.
4. Generates and saves 3 publication-quality Matplotlib figures to plots/.
"""

import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.arps_dca import run_arps_analysis
from src.ipr_skin import compute_skin_sensitivity
from src.nodal_operating_point import run_tubing_sensitivity


def plot_ipr_skin_sensitivity(ipr_data: Dict[str, Dict], save_path: str = "plots/ipr_skin_sensitivity.png") -> None:
    """Plot and save Composite Darcy-Vogel IPR curves for damaged, base, and stimulated skins."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    colors = {"Damaged Well (S = +5)": "#d9534f", "Zero-Skin Base (S = 0)": "#0275d8", "Stimulated Well (S = -3)": "#5cb85c"}
    styles = {"Damaged Well (S = +5)": "--", "Zero-Skin Base (S = 0)": "-", "Stimulated Well (S = -3)": "-."}

    pb_val = 2000.0
    for name, d in ipr_data.items():
        c = colors.get(name, "#333333")
        s = styles.get(name, "-")
        label = f"{name} (AOF = {d['aof']:.0f} BOPD, FE = {d['flow_efficiency']:.2f})"
        ax.plot(d["rate"], d["pwf"], label=label, color=c, linestyle=s, linewidth=2.2)
        ax.scatter([d["aof"]], [0.0], color=c, s=50, zorder=5)

    ax.axhline(pb_val, color="gray", linestyle=":", linewidth=1.5, label=f"Bubble Point (Pb = {pb_val:.0f} psi)")
    ax.set_title("Composite Darcy-Vogel IPR Curve: Skin Factor Sensitivity", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Oil Production Rate, q_o (STB/day)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Flowing Bottomhole Pressure, P_wf (psi)", fontsize=11, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_nodal_intersection(nodal_data: Dict, save_path: str = "plots/nodal_analysis_intersection.png") -> None:
    """Plot and save Nodal Analysis curves with operating points."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    q_grid = nodal_data["q_grid"]
    pwf_ipr = nodal_data["pwf_ipr"]

    ax.plot(q_grid, pwf_ipr, color="#111111", linewidth=2.5, label="Reservoir Inflow (IPR: S = 0)")

    tubing_colors = {"2.375\" (1.995\" ID)": "#f0ad4e", "2.875\" (2.441\" ID)": "#5bc0de", "3.500\" (2.992\" ID)": "#6f42c1"}

    for name, d in nodal_data["tubings"].items():
        c = tubing_colors.get(name, "#e83e8c")
        ax.plot(q_grid, d["vlp_curve"], label=f"VLP Tubing {name}", color=c, linewidth=2.0, linestyle="--")
        if d["q_op"] is not None:
            ax.scatter([d["q_op"]], [d["pwf_op"]], color=c, s=80, edgecolors="black", linewidth=1.2, zorder=6)
            ax.annotate(
                f"({d['q_op']:.0f} BOPD, {d['pwf_op']:.0f} psi)",
                xy=(d["q_op"], d["pwf_op"]),
                xytext=(d["q_op"] + 40, d["pwf_op"] + 60),
                fontsize=8.5,
                fontweight="bold",
                color=c,
                arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
            )

    ax.set_title("Nodal Analysis: Reservoir IPR & Tubing Intake (VLP) Intersection", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Flow Rate, q (STB/day)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pressure, P_wf (psi)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(q_grid) * 1.05)
    ax.set_ylim(bottom=2000, top=max(pwf_ipr) * 1.05)
    ax.legend(frameon=True, fontsize=9.5, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def plot_arps_decline_forecast(dca_data: Dict, save_path: str = "plots/arps_decline_forecast.png") -> None:
    """Plot historical production and fitted Arps decline curves out to economic limit."""
    df = dca_data["df"]
    fits = dca_data["fits"]
    q_econ = dca_data["q_econ"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    # Historical data scatter
    ax.scatter(df["Time_years"], df["Oil_Rate_bopd"], color="#1f77b4", s=35, alpha=0.85, label="Monthly History (36 mo)", zorder=4)

    # Forecast time array up to max economic limit (e.g. 25 years or max t_econ)
    max_t_plot = min(25.0, max(f["t_econ_years"] for f in fits.values() if f["b"] < 1.0) * 1.1)
    t_fcst = np.linspace(0.0833, max_t_plot, 300)

    colors = {"Exponential (b=0)": "#2ca02c", "Hyperbolic (0<b<1)": "#d62728", "Harmonic (b=1)": "#9467bd"}
    styles = {"Exponential (b=0)": "--", "Hyperbolic (0<b<1)": "-", "Harmonic (b=1)": "-."}

    for name, m in fits.items():
        q_fcst = m["model_func"](t_fcst)
        c = colors.get(name, "#333333")
        s = styles.get(name, "-")
        label = f"{name}: R²={m['r2']:.3f}, EUR={m['eur_stb']/1e6:.2f} MMSTB, t_econ={m['t_econ_years']:.1f} yr"
        ax.plot(t_fcst, q_fcst, label=label, color=c, linestyle=s, linewidth=2.0)

    ax.axhline(q_econ, color="crimson", linestyle=":", linewidth=1.5, label=f"Economic Limit (q_econ = {q_econ:.0f} BOPD)")
    ax.set_title("Arps Decline Curve Analysis: Historical Fit & Long-Term Forecast", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Oil Production Rate (STB/day)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max_t_plot)
    ax.set_ylim(0, max(df["Oil_Rate_bopd"]) * 1.15)
    ax.legend(frameon=True, fontsize=9.0, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


def main() -> None:
    """Run CLI workflows, display summary metrics, and save figures."""
    print("=" * 78)
    print(" INFLOW PERFORMANCE (IPR), NODAL ANALYSIS & ARPS DECLINE CURVE ANALYSIS TOOL")
    print(" Standard Undergraduate Petroleum Engineering Textbook Formulations")
    print("=" * 78)

    # 1. Composite Darcy-Vogel IPR and Skin Sensitivity
    print("\n[SECTION 1] COMPOSITE DARCY-VOGEL IPR & SKIN SENSITIVITY")
    print("-" * 78)
    ipr_results = compute_skin_sensitivity(pr_bar=3800.0, pb=2200.0, k_md=50.0, h_ft=50.0, mu_cp=1.2, bo_rb_stb=1.25, re_ft=1490.0, rw_ft=0.328)
    print(f"{'Condition / Skin Factor':<28} | {'J (STB/d/psi)':<13} | {'Flow Eff (FE)':<13} | {'q @ Pb (BOPD)':<13} | {'AOF (BOPD)':<10}")
    print("-" * 88)
    for name, d in ipr_results.items():
        print(f"{name:<28} | {d['J']:<13.4f} | {d['flow_efficiency']:<13.3f} | {d['q_b']:<13.1f} | {d['aof']:<10.1f}")
    plot_ipr_skin_sensitivity(ipr_results)

    # 2. Nodal Operating Point & Tubing Sensitivity
    print("\n[SECTION 2] NODAL ANALYSIS & VERTICAL LIFT PERFORMANCE (VLP)")
    print("-" * 78)
    j_base = ipr_results["Zero-Skin Base (S = 0)"]["J"]
    nodal_results = run_tubing_sensitivity(pr_bar=3800.0, pb=2200.0, j_index=j_base, pwh_psi=150.0, tvd_ft=8000.0, gamma_mix=0.85, mu_cp=1.2)
    print(f"{'Tubing Size (Nominal)':<22} | {'Tubing ID (in)':<14} | {'Operating Rate (BOPD)':<21} | {'Operating Pwf (psi)':<19}")
    print("-" * 88)
    for name, d in nodal_results["tubings"].items():
        rate_str = f"{d['q_op']:.1f}" if d["q_op"] is not None else "N/A"
        pwf_str = f"{d['pwf_op']:.1f}" if d["pwf_op"] is not None else "N/A"
        print(f"{name:<22} | {d['tubing_id_in']:<14.3f} | {rate_str:<21} | {pwf_str:<19}")
    plot_nodal_intersection(nodal_results)

    # 3. Arps Decline Curve Analysis & EUR
    print("\n[SECTION 3] ARPS DECLINE CURVE ANALYSIS & EUR FORECASTING")
    print("-" * 78)
    dca_results = run_arps_analysis("data/monthly_production_history.csv", q_econ=25.0)
    print(f"{'Arps Model':<22} | {'qi (BOPD)':<10} | {'Di (/year)':<11} | {'b':<6} | {'R² Fit':<8} | {'t_econ (yr)':<11} | {'EUR (MMSTB)':<11}")
    print("-" * 92)
    for model_name, m in dca_results["fits"].items():
        print(f"{model_name:<22} | {m['qi']:<10.1f} | {m['di']:<11.4f} | {m['b']:<6.2f} | {m['r2']:<8.4f} | {m['t_econ_years']:<11.2f} | {m['eur_stb']/1e6:<11.3f}")
    plot_arps_decline_forecast(dca_results)

    print("\n" + "=" * 78)
    print(" ALL CALCULATIONS COMPLETED SUCCESSFULLY. 3 PLOTS SAVED IN plots/ FOLDER.")
    print("=" * 78)


if __name__ == "__main__":
    main()

