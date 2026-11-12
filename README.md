# Inflow Performance (IPR), Nodal Analysis & Arps Decline Curve Analysis Tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![Petroleum Engineering](https://img.shields.io/badge/Field-Petroleum%20Production%20Engineering-darkgreen.svg)]()

A modular, textbook-rigorous Python engineering workbench for **Reservoir Inflow Performance (IPR)**, **Wellbore Tubing Intake / Vertical Lift Performance (VLP)**, **Nodal Operating Point Optimization**, and **Arps Decline Curve Analysis (DCA)**.

Validated against classical published petroleum literature solutions (Brown, Beggs, Guo et al.) and backtested on authentic open-source offshore production data from the **Equinor Volve Field (Well 15/9-F-12)**.

---

## Key Features

1. **Composite Darcy-Vogel Inflow Performance (IPR):**
   * Single-phase undersaturated radial Darcy flow for $P_{wf} \ge P_b$.
   * Two-phase saturated Vogel empirical correlation for $P_{wf} < P_b$ with $C^1$ derivative slope continuity at the bubble point:
     $$q_v = \frac{J \cdot P_b}{1.8}$$
   * Mechanical Skin Factor sensitivity ($S = +5$ damaged, $S = 0$ base, $S = -3$ stimulated) and Flow Efficiency ($FE$).
   * Analytical forward $q_o(P_{wf})$ and inverse $P_{wf}(q_o)$ formulations.

2. **Vertical Lift Performance (VLP) & Nodal Analysis:**
   * Liquid hydrostatic column gradient calculation:
     $$P_{\text{hydrostatic}} = 0.433 \cdot \gamma_{\text{mix}} \cdot TVD$$
   * Darcy-Weisbach frictional pressure loss using the explicit Haaland friction factor correlation across standard API tubing sizes ($2.375''$, $2.875''$, $3.500''$).
   * High-precision numerical determination of the sandface Nodal Operating Point $(q_{\text{op}}, P_{wf,\text{op}})$ via `scipy.optimize.brentq`.

3. **Arps Decline Curve Analysis & EUR Estimation:**
   * Classical Arps decline families:
     * **Exponential ($b = 0$):** $q(t) = q_i \exp(-D_i t)$
     * **Hyperbolic ($0 < b < 1$):** $q(t) = \frac{q_i}{(1 + b D_i t)^{1/b}}$
     * **Harmonic ($b = 1$):** $q(t) = \frac{q_i}{1 + D_i t}$
   * Non-linear regression fitting via `scipy.optimize.curve_fit` with $R^2$ goodness-of-fit metrics.
   * Long-term forecasting out to economic limit rate ($q_{\text{econ}}$) with cumulative production ($N_p$) and Estimated Ultimate Recovery (EUR).

4. **Real Field Benchmarking & Validation:**
   * Verified against published textbook benchmark test problems ($0.00\%$ discrepancy).
   * 70/30 train/test out-of-sample hindcasting on 71 months of continuous production from North Sea Well 15/9-F-12.

5. **Interactive Web Dashboard:**
   * Full-featured Streamlit dashboard (`app.py`) providing real-time sliders for reservoir pressure, bubble point, skin factor, wellhead pressure, TVD, fluid gravity, and economic limit.

---

## Project Structure

```text
production-ipr-dca/
├── data/
│   ├── monthly_production_history.csv       # 36-month baseline production history
│   └── real_field_benchmarks/
│       ├── real_volve_15_9_F12_monthly.csv  # Equinor Volve Well 15/9-F-12 monthly rates
│       └── real_volve_production.csv        # Multi-well Volve field historical dataset
├── src/
│   ├── ipr_skin.py                          # Composite Darcy-Vogel IPR & Skin sensitivity
│   ├── nodal_operating_point.py             # Tubing intake (VLP) & Brent's method solver
│   └── arps_dca.py                          # Arps decline models, fitting & EUR estimation
├── plots/
│   ├── ipr_skin_sensitivity.png             # Darcy-Vogel IPR curves for S = +5, 0, -3
│   ├── nodal_analysis_intersection.png      # IPR vs VLP intersection operating points
│   ├── arps_decline_forecast.png            # Arps fit & long-term forecast to q_econ
│   └── real_volve_dca_hindcast.png          # 70/30 train/test hindcast on Volve 15/9-F-12
├── main.py                                  # CLI runner for calculations, tables & plots
├── verify_real_data.py                      # Automated benchmark verification & hindcasting
├── app.py                                   # Streamlit interactive engineering dashboard
├── VALIDATION_REPORT.md                     # Comprehensive benchmark verification report
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## Installation & Setup

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd production-ipr-dca
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run Core Calculations & Generate Plots (CLI)
Executes IPR sensitivity, Nodal tubing analysis, and Arps DCA curve fitting, printing summary tables and saving high-resolution figures to `plots/`:
```bash
python main.py
```

### 2. Run Benchmark Verification & Field Hindcast
Executes Test 1 (textbook solution comparison) and Test 2 (70/30 train-test hindcast on Equinor Volve 15/9-F-12):
```bash
python verify_real_data.py
```

### 3. Launch the Interactive Streamlit Web App
Starts the browser-based engineering dashboard with interactive sliders:
```bash
streamlit run app.py
```

---

## Mathematical Formulations

### 1. Radial Darcy Flow (Undersaturated Reservoir, $P_{wf} \ge P_b$)
$$J = \frac{q_o}{\bar{P}_r - P_{wf}} = \frac{0.00708 \times k \times h}{\mu_o \times B_o \times \left[\ln\left(\frac{r_e}{r_w}\right) - 0.75 + S\right]}$$

### 2. Vogel Correlation (Saturated Reservoir, $P_{wf} < P_b$)
$$q_b = J \cdot (\bar{P}_r - P_b)$$
$$q_v = \frac{J \cdot P_b}{1.8}$$
$$q_o = q_b + q_v \left[1 - 0.2\left(\frac{P_{wf}}{P_b}\right) - 0.8\left(\frac{P_{wf}}{P_b}\right)^2\right]$$
$$AOF = q_b + q_v$$

### 3. Tubing Intake Pressure (VLP)
$$P_{wf,\text{VLP}}(q) = P_{wh} + P_{\text{hydrostatic}} + \Delta P_{\text{friction}}(q)$$
$$\Delta P_{\text{friction}} = \frac{f \cdot \rho \cdot v^2 \cdot L}{772.18 \cdot d}$$
where fluid velocity $v = \frac{0.01193 \cdot q}{d^2}\text{ ft/s}$ and $f$ is evaluated via the Haaland explicit correlation.

### 4. Arps Decline Equations & EUR
* **Exponential ($b = 0$):**
  $$EUR = 365.25 \times \frac{q_i - q_{\text{econ}}}{D_i}$$
* **Hyperbolic ($0 < b < 1$):**
  $$EUR = 365.25 \times \frac{q_i^b}{(1 - b) D_i} \left[q_i^{1-b} - q_{\text{econ}}^{1-b}\right]$$
* **Harmonic ($b = 1$):**
  $$EUR = 365.25 \times \frac{q_i}{D_i} \ln\left(\frac{q_i}{q_{\text{econ}}}\right)$$

*(Note: The $365.25$ factor converts daily rates in STB/day to annual cumulative barrels when nominal decline $D_i$ is expressed in $\text{year}^{-1}$.)*

---

## Benchmark Verification Summary

| Metric | Published Reference | Code Output | Abs Error (%) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Productivity Index $J$** | $0.9229\text{ STB/d/psi}$ | $0.9229\text{ STB/d/psi}$ | $0.0026\%$ | **PASS** |
| **Rate at Bubble Point $q_b$** | $830.63\text{ BOPD}$ | $830.63\text{ BOPD}$ | $0.0002\%$ | **PASS** |
| **Vogel Potential $q_v$** | $1,076.74\text{ BOPD}$ | $1,076.74\text{ BOPD}$ | $0.0004\%$ | **PASS** |
| **Absolute Open Flow $AOF$** | $1,907.38\text{ BOPD}$ | $1,907.38\text{ BOPD}$ | $0.0002\%$ | **PASS** |
| **Tubing Hydrostatic Head $P_{\text{hydro}}$** | $1,840.25\text{ psi}$ | $1,840.25\text{ psi}$ | $0.0000\%$ | **PASS** |
| **Nodal Operating Rate $q_{\text{op}}$** | $919.10\text{ BOPD}$ | $919.09\text{ BOPD}$ | $0.0014\%$ | **PASS** |
| **Nodal Operating Pressure $P_{wf,\text{op}}$** | $2,002.13\text{ psi}$ | $2,002.13\text{ psi}$ | $0.0000\%$ | **PASS** |

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for full backtesting details, out-of-sample metrics ($R^2$, RMSE, MAPE), and reservoir engineering observations on North Sea waterflood dynamics.

---

## License

This project is licensed under the [MIT License](LICENSE).
