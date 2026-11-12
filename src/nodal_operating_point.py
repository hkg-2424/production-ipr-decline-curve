"""
src/nodal_operating_point.py
----------------------------
Tubing Intake (Vertical Lift Performance - VLP) & Nodal Operating Point.

Textbook Equations Implemented:
1. Hydrostatic Head: P_hydro = 0.433 * gamma_mix * TVD
2. Darcy-Weisbach Friction:
       v = 0.01193 * q / d^2  (ft/s)
       Re = 1.48 * q * rho / (d * mu)
       Haaland friction factor: 1/sqrt(f) = -1.8 * log10([(eps/d)/3.7]^1.11 + 6.9/Re)
       Delta_P_friction = (f * rho * v^2 * L) / (772.18 * d)  (psi)
3. Total Tubing Intake Pressure:
       P_wf_VLP(q) = P_wh + P_hydrostatic + Delta_P_friction(q)
4. Numerical Nodal Operating Point:
       Find q_op where P_wf_IPR(q) - P_wf_VLP(q) = 0 using scipy.optimize.brentq.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import brentq

import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import brentq

# Ensure parent directory is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.ipr_skin import inverse_composite_ipr_pressure, calculate_productivity_index
except ImportError:
    from ipr_skin import inverse_composite_ipr_pressure, calculate_productivity_index


# Standard API Tubing Specifications (Outer Diameter -> Nominal Inner Diameter in inches)
API_TUBING_SIZES: Dict[str, float] = {
    '2.375" (1.995" ID)': 1.995,
    '2.875" (2.441" ID)': 2.441,
    '3.500" (2.992" ID)': 2.992,
}


def calculate_hydrostatic_pressure(gamma_mix: float, tvd_ft: float) -> float:
    r"""
    Calculate static liquid hydrostatic pressure gradient and pressure drop.

    Textbook Equation:
        P_hydrostatic = 0.433 * gamma_mix * TVD

    Units:
        gamma_mix: Fluid specific gravity relative to fresh water (water = 1.0)
        tvd_ft: True Vertical Depth (ft)
        Water gradient = 0.433 psi/ft
        Output: Hydrostatic bottomhole pressure contribution (psi)
    """
    return float(0.433 * gamma_mix * tvd_ft)


def calculate_friction_pressure_drop(
    qo_bopd: Union[float, np.ndarray],
    tubing_id_in: float,
    length_ft: float,
    gamma_mix: float = 0.85,
    mu_cp: float = 1.2,
    roughness_in: float = 0.0006,
) -> Union[float, np.ndarray]:
    r"""
    Calculate friction pressure drop in tubing using Darcy-Weisbach and Haaland correlation.

    Textbook Equations:
        1. Average velocity: v = (q * 5.615 / 86400) / (pi * (d/12)^2 / 4) = 0.01193 * q / d^2  (ft/s)
        2. Density: rho = 62.4 * gamma_mix  (lbm/ft^3)
        3. Reynolds number: Re = (1.48 * q * rho) / (d * mu)
        4. Darcy friction factor:
           - Laminar (Re < 2100): f = 64 / Re
           - Turbulent (Re >= 2100, Haaland equation):
             1 / sqrt(f) = -1.8 * log10( ((eps/d)/3.7)^1.11 + 6.9 / Re )
        5. Frictional pressure loss (psi):
           Delta_P_f = (f * rho * v^2 * L) / (2 * gc * (d/12) * 144) = (f * rho * v^2 * L) / (772.18 * d)
    """
    q_arr = np.atleast_1d(np.asarray(qo_bopd, dtype=float))
    rho = 62.4 * gamma_mix  # fluid density in lbm/ft^3
    d = tubing_id_in

    # Velocity in ft/sec (zero rate -> zero velocity)
    v = 0.01193 * q_arr / (d ** 2)

    # Reynolds number
    # Re = rho * v * (d/12) / (mu * 6.7197e-4) = (1.48 * q * rho) / (d * mu)
    re = np.where(q_arr > 1e-4, (1.48 * q_arr * rho) / (d * mu_cp), 1.0)

    # Haaland explicit friction factor
    rel_roughness = roughness_in / d
    f_turb = (-1.8 * np.log10(((rel_roughness / 3.7) ** 1.11) + (6.9 / np.maximum(re, 1.0)))) ** -2.0
    f_lam = 64.0 / np.maximum(re, 1.0)
    f = np.where(re < 2100.0, f_lam, f_turb)

    # Darcy-Weisbach field formula for Delta P in psi:
    # Delta_P_f = (f * rho * v^2 * L) / (772.18 * d)
    delta_p_f = (f * rho * (v ** 2) * length_ft) / (772.18 * d)
    delta_p_f = np.where(q_arr <= 1e-4, 0.0, delta_p_f)

    if np.isscalar(qo_bopd):
        return float(delta_p_f[0])
    return delta_p_f


def tubing_intake_vlp_pressure(
    qo_bopd: Union[float, np.ndarray],
    pwh_psi: float,
    tvd_ft: float,
    tubing_id_in: float,
    gamma_mix: float = 0.85,
    mu_cp: float = 1.2,
) -> Union[float, np.ndarray]:
    r"""
    Compute total flowing bottomhole pressure required by the tubing (VLP curve):
        P_wf_VLP(q) = P_wh + P_hydrostatic + Delta_P_friction(q)
    """
    p_hydro = calculate_hydrostatic_pressure(gamma_mix, tvd_ft)
    p_fric = calculate_friction_pressure_drop(qo_bopd, tubing_id_in, tvd_ft, gamma_mix, mu_cp)
    return pwh_psi + p_hydro + p_fric


def find_nodal_operating_point(
    pr_bar: float,
    pb: float,
    j_index: float,
    pwh_psi: float,
    tvd_ft: float,
    tubing_id_in: float,
    gamma_mix: float = 0.85,
    mu_cp: float = 1.2,
) -> Tuple[Optional[float], Optional[float], str]:
    r"""
    Find the Nodal Operating Point (q_op, Pwf_op) by solving:
        F(q) = P_wf_IPR(q) - P_wf_VLP(q) = 0
    using scipy.optimize.brentq.
    """
    # Absolute open flow
    pb_eff = min(pb, pr_bar)
    q_b = j_index * (pr_bar - pb_eff) if pr_bar > pb_eff else 0.0
    q_v = (j_index * pb_eff) / 1.8 if pb_eff > 0 else 0.0
    aof = q_b + q_v

    # Evaluate static condition (q = 0)
    p_wf_ipr_0 = pr_bar
    p_wf_vlp_0 = tubing_intake_vlp_pressure(0.0, pwh_psi, tvd_ft, tubing_id_in, gamma_mix, mu_cp)

    if p_wf_ipr_0 <= p_wf_vlp_0:
        return (
            None,
            None,
            f"No natural flow: Reservoir pressure ({pr_bar:.1f} psi) <= Static Head ({p_wf_vlp_0:.1f} psi). Artificial lift required.",
        )

    def objective(q_val: float) -> float:
        pwf_ipr = inverse_composite_ipr_pressure(q_val, pr_bar, pb, j_index)
        pwf_vlp = tubing_intake_vlp_pressure(q_val, pwh_psi, tvd_ft, tubing_id_in, gamma_mix, mu_cp)
        return float(pwf_ipr - pwf_vlp)

    # Check bounds for brentq
    f_low = objective(1e-2)
    f_high = objective(aof - 1e-2)

    if f_low * f_high > 0:
        # VLP does not intersect IPR within flowable AOF limit
        return (
            None,
            None,
            "No intersection within valid flow range [0, AOF]. Well cannot support this tubing size naturally.",
        )

    # Solve root
    q_op = float(brentq(objective, 1e-2, aof - 1e-2, xtol=1e-4, rtol=1e-4))
    pwf_op = float(tubing_intake_vlp_pressure(q_op, pwh_psi, tvd_ft, tubing_id_in, gamma_mix, mu_cp))

    return q_op, pwf_op, "Flowing naturally at stable operating point."


def run_tubing_sensitivity(
    pr_bar: float = 3800.0,
    pb: float = 2200.0,
    j_index: float = 1.5,
    pwh_psi: float = 150.0,
    tvd_ft: float = 8000.0,
    gamma_mix: float = 0.85,
    mu_cp: float = 1.2,
    tubing_dict: Optional[Dict[str, float]] = None,
    num_points: int = 100,
) -> Dict:
    """Run Nodal Analysis across multiple standard tubing sizes."""
    if tubing_dict is None:
        tubing_dict = API_TUBING_SIZES

    # Flow rates up to AOF
    pb_eff = min(pb, pr_bar)
    q_b = j_index * (pr_bar - pb_eff) if pr_bar > pb_eff else 0.0
    q_v = (j_index * pb_eff) / 1.8 if pb_eff > 0 else 0.0
    aof = q_b + q_v

    q_grid = np.linspace(0.0, aof, num_points)
    pwf_ipr = inverse_composite_ipr_pressure(q_grid, pr_bar, pb, j_index)

    tubing_results = {}
    for label, id_in in tubing_dict.items():
        vlp_curve = tubing_intake_vlp_pressure(q_grid, pwh_psi, tvd_ft, id_in, gamma_mix, mu_cp)
        q_op, pwf_op, status = find_nodal_operating_point(
            pr_bar, pb, j_index, pwh_psi, tvd_ft, id_in, gamma_mix, mu_cp
        )
        tubing_results[label] = {
            'tubing_id_in': id_in,
            'vlp_curve': vlp_curve,
            'q_op': q_op,
            'pwf_op': pwf_op,
            'status': status,
        }

    return {
        'q_grid': q_grid,
        'pwf_ipr': pwf_ipr,
        'aof': aof,
        'tubings': tubing_results,
    }


if __name__ == "__main__":
    res = run_tubing_sensitivity()
    print("=== Nodal Analysis Operating Points ===")
    for name, d in res['tubings'].items():
        if d['q_op'] is not None:
            print(f"{name}: Operating Rate = {d['q_op']:.1f} BOPD, Pwf = {d['pwf_op']:.1f} psi")
        else:
            print(f"{name}: {d['status']}")
