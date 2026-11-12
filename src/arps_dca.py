"""
src/arps_dca.py
---------------
Arps Decline Curve Analysis (DCA), Model Fitting, Forecasting & EUR Estimation.

Textbook Equations Implemented:
1. Exponential Decline (b = 0):
       q(t) = q_i * exp(-D_i * t)
       Np(t) = [q_i - q(t)] / D_i
       EUR = (q_i - q_econ) / D_i

2. Hyperbolic Decline (0 < b < 1):
       q(t) = q_i / (1 + b * D_i * t)^(1/b)
       EUR = [q_i^b / ((1 - b) * D_i)] * (q_i^(1 - b) - q_econ^(1 - b))

3. Harmonic Decline (b = 1):
       q(t) = q_i / (1 + D_i * t)
       EUR = (q_i / D_i) * ln(q_i / q_econ)

* Note on field units:
  When rates are in BOPD (STB/day) and t is in years, D_i is in year^(-1).
  Cumulative volume in STB is obtained by multiplying by 365.25 days/year.
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def arps_exponential_rate(t_years: Union[float, np.ndarray], qi: float, di: float) -> Union[float, np.ndarray]:
    r"""
    Exponential decline equation (b = 0):
        q(t) = q_i * exp(-D_i * t)
    """
    return qi * np.exp(-di * t_years)


def arps_hyperbolic_rate(t_years: Union[float, np.ndarray], qi: float, di: float, b: float) -> Union[float, np.ndarray]:
    r"""
    Hyperbolic decline equation (0 < b < 1):
        q(t) = q_i / (1 + b * D_i * t)^(1/b)
    """
    # Guard against non-physical negative values inside bracket
    base = np.maximum(1.0 + b * di * t_years, 1e-6)
    return qi / (base ** (1.0 / b))


def arps_harmonic_rate(t_years: Union[float, np.ndarray], qi: float, di: float) -> Union[float, np.ndarray]:
    r"""
    Harmonic decline equation (b = 1):
        q(t) = q_i / (1 + D_i * t)
    """
    base = np.maximum(1.0 + di * t_years, 1e-6)
    return qi / base


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 (coefficient of determination) goodness-of-fit."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - (ss_res / ss_tot))


def calculate_time_to_economic_limit(
    qi: float,
    di: float,
    q_econ: float,
    b: float = 0.0,
) -> float:
    r"""
    Calculate time (years) to reach economic limit rate:
        Exponential (b = 0): t_econ = ln(q_i / q_econ) / D_i
        Hyperbolic (0 < b < 1): t_econ = [(q_i / q_econ)^b - 1] / (b * D_i)
        Harmonic (b = 1): t_econ = (q_i / q_econ - 1) / D_i
    """
    if qi <= q_econ or di <= 0:
        return 0.0

    if b == 0.0:
        # Exponential: t = ln(qi / q_econ) / di
        return float(np.log(qi / q_econ) / di)
    elif abs(b - 1.0) < 1e-5:
        # Harmonic: t = (qi / q_econ - 1) / di
        return float((qi / q_econ - 1.0) / di)
    else:
        # Hyperbolic: t = [(qi / q_econ)^b - 1] / (b * di)
        return float(((qi / q_econ) ** b - 1.0) / (b * di))


def calculate_eur(
    qi: float,
    di: float,
    q_econ: float,
    b: float = 0.0,
    days_per_year: float = 365.25,
) -> float:
    r"""
    Calculate Estimated Ultimate Recovery (EUR) in Stock Tank Barrels (STB).

    Textbook Equations:
        Exponential (b = 0):
            EUR_exp = days_per_year * (q_i - q_econ) / D_i

        Hyperbolic (0 < b < 1):
            EUR_hyp = days_per_year * [q_i^b / ((1 - b) * D_i)] * [q_i^(1 - b) - q_econ^(1 - b)]

        Harmonic (b = 1):
            EUR_harm = days_per_year * (q_i / D_i) * ln(q_i / q_econ)
    """
    if qi <= q_econ or di <= 0:
        return 0.0

    if b == 0.0:
        # EUR_exp = (qi - q_econ) / Di
        eur_daily = (qi - q_econ) / di
    elif abs(b - 1.0) < 1e-5:
        # EUR_harm = (qi / di) * ln(qi / q_econ)
        eur_daily = (qi / di) * np.log(qi / q_econ)
    else:
        # EUR_hyp = [qi^b / ((1 - b) * di)] * [qi^(1 - b) - q_econ^(1 - b)]
        eur_daily = (qi ** b / ((1.0 - b) * di)) * (qi ** (1.0 - b) - q_econ ** (1.0 - b))

    return float(days_per_year * eur_daily)


def fit_decline_curves(
    time_years: np.ndarray,
    rates_bopd: np.ndarray,
    q_econ: float = 25.0,
) -> Dict[str, Dict]:
    """
    Fit Exponential, Hyperbolic, and Harmonic Arps decline curves to production data
    using scipy.optimize.curve_fit.
    """
    t_data = np.asarray(time_years, dtype=float)
    q_data = np.asarray(rates_bopd, dtype=float)

    # Initial parameter guesses
    qi_guess = float(q_data[0])
    di_guess = 0.5  # 50% nominal annual decline

    results = {}

    # 1. Fit Exponential Model (b = 0)
    popt_exp, _ = curve_fit(
        arps_exponential_rate,
        t_data,
        q_data,
        p0=[qi_guess, di_guess],
        bounds=([0.0, 1e-4], [qi_guess * 3.0, 5.0]),
    )
    qi_exp, di_exp = popt_exp
    q_pred_exp = arps_exponential_rate(t_data, qi_exp, di_exp)
    r2_exp = calculate_r_squared(q_data, q_pred_exp)
    t_econ_exp = calculate_time_to_economic_limit(qi_exp, di_exp, q_econ, b=0.0)
    eur_exp = calculate_eur(qi_exp, di_exp, q_econ, b=0.0)

    results['Exponential (b=0)'] = {
        'qi': qi_exp,
        'di': di_exp,
        'b': 0.0,
        'r2': r2_exp,
        't_econ_years': t_econ_exp,
        'eur_stb': eur_exp,
        'model_func': lambda t: arps_exponential_rate(t, qi_exp, di_exp),
    }

    # 2. Fit Hyperbolic Model (0 < b < 1)
    popt_hyp, _ = curve_fit(
        arps_hyperbolic_rate,
        t_data,
        q_data,
        p0=[qi_guess, di_guess, 0.5],
        bounds=([0.0, 1e-4, 0.01], [qi_guess * 3.0, 5.0, 0.99]),
    )
    qi_hyp, di_hyp, b_hyp = popt_hyp
    q_pred_hyp = arps_hyperbolic_rate(t_data, qi_hyp, di_hyp, b_hyp)
    r2_hyp = calculate_r_squared(q_data, q_pred_hyp)
    t_econ_hyp = calculate_time_to_economic_limit(qi_hyp, di_hyp, q_econ, b=b_hyp)
    eur_hyp = calculate_eur(qi_hyp, di_hyp, q_econ, b=b_hyp)

    results['Hyperbolic (0<b<1)'] = {
        'qi': qi_hyp,
        'di': di_hyp,
        'b': b_hyp,
        'r2': r2_hyp,
        't_econ_years': t_econ_hyp,
        'eur_stb': eur_hyp,
        'model_func': lambda t: arps_hyperbolic_rate(t, qi_hyp, di_hyp, b_hyp),
    }

    # 3. Fit Harmonic Model (b = 1)
    popt_harm, _ = curve_fit(
        arps_harmonic_rate,
        t_data,
        q_data,
        p0=[qi_guess, di_guess],
        bounds=([0.0, 1e-4], [qi_guess * 3.0, 5.0]),
    )
    qi_harm, di_harm = popt_harm
    q_pred_harm = arps_harmonic_rate(t_data, qi_harm, di_harm)
    r2_harm = calculate_r_squared(q_data, q_pred_harm)
    t_econ_harm = calculate_time_to_economic_limit(qi_harm, di_harm, q_econ, b=1.0)
    eur_harm = calculate_eur(qi_harm, di_harm, q_econ, b=1.0)

    results['Harmonic (b=1)'] = {
        'qi': qi_harm,
        'di': di_harm,
        'b': 1.0,
        'r2': r2_harm,
        't_econ_years': t_econ_harm,
        'eur_stb': eur_harm,
        'model_func': lambda t: arps_harmonic_rate(t, qi_harm, di_harm),
    }

    return results


def run_arps_analysis(csv_path: str = "data/monthly_production_history.csv", q_econ: float = 25.0) -> Dict:
    """Load monthly history CSV and run complete decline curve analysis."""
    df = pd.read_csv(csv_path)
    t_data = df["Time_years"].values
    q_data = df["Oil_Rate_bopd"].values

    fits = fit_decline_curves(t_data, q_data, q_econ=q_econ)
    return {"df": df, "fits": fits, "q_econ": q_econ}


if __name__ == "__main__":
    res = run_arps_analysis()
    print("=== Arps Decline Curve Analysis Results ===")
    for model_name, metrics in res['fits'].items():
        print(f"\n{model_name}:")
        print(f"  qi = {metrics['qi']:.1f} BOPD | Di = {metrics['di']:.4f}/yr | b = {metrics['b']:.2f}")
        print(f"  R^2 = {metrics['r2']:.4f}")
        print(f"  Time to {res['q_econ']} BOPD: {metrics['t_econ_years']:.2f} years")
        print(f"  Estimated Ultimate Recovery (EUR): {metrics['eur_stb']/1e6:.3f} MMSTB")

