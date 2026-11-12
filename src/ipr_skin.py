"""
src/ipr_skin.py
---------------
Reservoir Inflow Performance Relationship (IPR) & Skin Factor Sensitivity.

Implements standard undergraduate petroleum engineering equations:
1. Radial Darcy Flow for undersaturated reservoirs (Pwf >= Pb)
2. Vogel's (1968) empirical correlation for saturated reservoirs (Pwf < Pb)
3. Composite Darcy-Vogel IPR formulation with slope continuity
4. Skin factor sensitivity (S = +5, S = 0, S = -3) and Flow Efficiency (FE)
"""

from typing import Dict, Tuple, Union
import numpy as np


def calculate_productivity_index(
    k_md: float,
    h_ft: float,
    mu_cp: float,
    bo_rb_stb: float,
    re_ft: float,
    rw_ft: float,
    skin: float = 0.0,
) -> float:
    r"""
    Calculate the Productivity Index (J) for steady/pseudo-steady radial Darcy flow.

    Textbook Equation:
        J = (0.00708 * k * h) / (mu_o * B_o * [ln(r_e / r_w) - 0.75 + S])

    Units:
        k: Permeability (mD)
        h: Net pay thickness (ft)
        mu_o: Oil viscosity (cP)
        B_o: Formation volume factor (rb/STB)
        r_e: Reservoir drainage radius (ft)
        r_w: Wellbore radius (ft)
        S: Van Everdingen & Hurst mechanical skin factor (dimensionless)
        Output J: Productivity Index (STB/day/psi)
    """
    # Denominator geometric & skin factor term: ln(re / rw) - 0.75 + S
    denom_term = np.log(re_ft / rw_ft) - 0.75 + skin
    if denom_term <= 0:
        raise ValueError(
            f"Skin factor S={skin} results in non-physical non-positive denominator: {denom_term:.3f}"
        )

    # Formula: J = (0.00708 * k * h) / (mu * Bo * [ln(re/rw) - 0.75 + S])
    j_index = (0.00708 * k_md * h_ft) / (mu_cp * bo_rb_stb * denom_term)
    return float(j_index)


def composite_darcy_vogel_rate(
    pwf: Union[float, np.ndarray],
    pr_bar: float,
    pb: float,
    j_index: float,
) -> Union[float, np.ndarray]:
    r"""
    Compute oil flow rate (qo) for any flowing bottomhole pressure (Pwf)
    using the Composite Darcy-Vogel model.

    Textbook Equations:
        For Pwf >= Pb (Single-phase undersaturated Darcy flow):
            q_o = J * (Pr_bar - Pwf)

        For Pwf < Pb (Two-phase saturated Vogel flow below bubble point):
            q_b = J * (Pr_bar - Pb)
            q_v = (J * Pb) / 1.8
            q_o = q_b + q_v * [1 - 0.2 * (Pwf / Pb) - 0.8 * (Pwf / Pb)^2]
    """
    pwf_arr = np.atleast_1d(np.asarray(pwf, dtype=float))
    # Saturated reservoir check: if Pr <= Pb, reservoir is saturated throughout
    pb_eff = min(pb, pr_bar)

    # 1. Flow rate at bubble point: q_b = J * (Pr_bar - Pb_eff)
    q_b = j_index * (pr_bar - pb_eff) if pr_bar > pb_eff else 0.0

    # 2. Maximum potential flow contribution below bubble point: q_v = (J * Pb_eff) / 1.8
    q_v = (j_index * pb_eff) / 1.8 if pb_eff > 0 else 0.0

    # Compute rate piecewise
    qo = np.zeros_like(pwf_arr)

    # Undersaturated segment (Pwf >= Pb_eff)
    mask_above = pwf_arr >= pb_eff
    qo[mask_above] = j_index * (pr_bar - pwf_arr[mask_above])

    # Saturated segment (Pwf < Pb_eff)
    mask_below = ~mask_above
    pwf_ratio = np.clip(pwf_arr[mask_below] / (pb_eff if pb_eff > 0 else 1.0), 0.0, 1.0)
    # Vogel equation: q_v * [1 - 0.2*(Pwf/Pb) - 0.8*(Pwf/Pb)^2]
    qo[mask_below] = q_b + q_v * (1.0 - 0.2 * pwf_ratio - 0.8 * (pwf_ratio ** 2))

    # Flow rate cannot be negative when Pwf >= Pr
    qo = np.maximum(qo, 0.0)

    if np.isscalar(pwf):
        return float(qo[0])
    return qo


def inverse_composite_ipr_pressure(
    qo: Union[float, np.ndarray],
    pr_bar: float,
    pb: float,
    j_index: float,
) -> Union[float, np.ndarray]:
    r"""
    Compute flowing bottomhole pressure (Pwf) from given flow rate (qo).
    Used directly in Nodal Analysis to construct the IPR curve as Pwf_IPR(q).

    Textbook Inverse Equations:
        q_b = J * (Pr_bar - Pb)
        q_v = (J * Pb) / 1.8
        AOF = q_b + q_v

        If qo <= q_b:
            Pwf = Pr_bar - qo / J

        If q_b < qo <= AOF:
            Let c = 1 - (qo - q_b) / q_v
            Solve 0.8 * (Pwf/Pb)^2 + 0.2 * (Pwf/Pb) - c = 0
            Pwf = Pb * (-0.2 + sqrt(0.04 + 3.2 * c)) / 1.6
    """
    q_arr = np.atleast_1d(np.asarray(qo, dtype=float))
    pb_eff = min(pb, pr_bar)
    q_b = j_index * (pr_bar - pb_eff) if pr_bar > pb_eff else 0.0
    q_v = (j_index * pb_eff) / 1.8 if pb_eff > 0 else 0.0
    aof = q_b + q_v

    pwf = np.zeros_like(q_arr)

    # Segment 1: Darcy region (qo <= q_b)
    mask_darcy = q_arr <= q_b
    pwf[mask_darcy] = pr_bar - (q_arr[mask_darcy] / j_index)

    # Segment 2: Vogel region (q_b < qo <= aof)
    mask_vogel = (~mask_darcy) & (q_arr <= aof)
    if np.any(mask_vogel) and q_v > 0:
        c = 1.0 - (q_arr[mask_vogel] - q_b) / q_v
        c = np.clip(c, 0.0, 1.0)
        # Quadratic formula root for 0.8 x^2 + 0.2 x - c = 0
        x = (-0.2 + np.sqrt(0.04 + 3.2 * c)) / 1.6
        pwf[mask_vogel] = pb_eff * np.clip(x, 0.0, 1.0)

    # Segment 3: Above AOF -> Pwf clamped to 0.0
    mask_above = q_arr > aof
    pwf[mask_above] = 0.0

    if np.isscalar(qo):
        return float(pwf[0])
    return pwf


def flow_efficiency(re_ft: float, rw_ft: float, skin: float) -> float:
    r"""
    Calculate Flow Efficiency (FE = J_actual / J_ideal):
        FE = [ln(re / rw) - 0.75] / [ln(re / rw) - 0.75 + S]
    """
    ideal_term = np.log(re_ft / rw_ft) - 0.75
    actual_term = ideal_term + skin
    return float(ideal_term / actual_term)


def compute_skin_sensitivity(
    pr_bar: float = 3000.0,
    pb: float = 2000.0,
    k_md: float = 50.0,
    h_ft: float = 50.0,
    mu_cp: float = 1.2,
    bo_rb_stb: float = 1.25,
    re_ft: float = 1490.0,
    rw_ft: float = 0.328,
    skin_cases: Tuple[float, ...] = (5.0, 0.0, -3.0),
    num_points: int = 100,
) -> Dict[str, Dict]:
    """
    Generate complete IPR curves and engineering metrics for specified skin cases:
    - Damaged Well (e.g., S = +5.0)
    - Zero-Skin Base Case (S = 0.0)
    - Stimulated/Acidized Well (e.g., S = -3.0)
    """
    results = {}
    pwf_grid = np.linspace(pr_bar, 0.0, num_points)

    labels = {
        5.0: "Damaged Well (S = +5)",
        0.0: "Zero-Skin Base (S = 0)",
        -3.0: "Stimulated Well (S = -3)",
    }

    for s in skin_cases:
        j = calculate_productivity_index(k_md, h_ft, mu_cp, bo_rb_stb, re_ft, rw_ft, skin=s)
        pb_eff = min(pb, pr_bar)
        q_b = j * (pr_bar - pb_eff) if pr_bar > pb_eff else 0.0
        q_v = (j * pb_eff) / 1.8 if pb_eff > 0 else 0.0
        aof = q_b + q_v
        fe = flow_efficiency(re_ft, rw_ft, skin=s)
        q_curve = composite_darcy_vogel_rate(pwf_grid, pr_bar, pb, j)

        name = labels.get(s, f"Skin S = {s:+.1f}")
        results[name] = {
            "skin": s,
            "J": j,
            "q_b": q_b,
            "q_v": q_v,
            "aof": aof,
            "flow_efficiency": fe,
            "pwf": pwf_grid,
            "rate": q_curve,
        }

    return results


if __name__ == "__main__":
    # Self-test when run directly
    sens = compute_skin_sensitivity()
    print("=== Composite Darcy-Vogel Skin Sensitivity Test ===")
    for case, data in sens.items():
        print(f"\n{case}:")
        print(f"  J: {data['J']:.4f} STB/d/psi | Flow Efficiency: {data['flow_efficiency']:.3f}")
        print(f"  Rate @ Pb: {data['q_b']:.1f} BOPD | AOF: {data['aof']:.1f} BOPD")

