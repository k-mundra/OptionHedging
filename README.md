# HedgeStorm: Quantitative Risk Management During Market Shocks with Delta and Vega Hedging in a Heston-Based Framework

Modeling & Simulation Course Project

### Group 19 - Kashvi Mundra, Aishwarya Pendyala, Anusha Saha, Soumil Sahu

[Link to overleaf doc (READ ONLY)](https://www.overleaf.com/read/xhndsyqxvgyj#ab3d43)

---

HedgeStorm is a Python-based simulation framework for evaluating hedging strategies under stochastic volatility, particularly during market shocks such as COVID-19.

## Overview

This project models option pricing using the **Heston stochastic volatility model**, and evaluates **delta** and **delta-vega** hedging strategies. We explore hedging effectiveness under both normal and volatile market conditions, using historical Apple options data.

Key components:
- **Model calibration** using a semi-analytical Heston solution
- **Monte Carlo simulations** to estimate PnL for hedged portfolios
- **Dynamic hedging strategies** with sensitivity to both price (delta) and volatility (vega)

## Project Structure

- `datasets/`
  - `AAPL_option_data.csv` – Apple options data used for calibration and simulation

- `modeling/`
  - `calibration.py` – Calibrate Heston model parameters using historical market data
  - `delta_hedge_mc.py` – Delta-only hedging strategy using Monte Carlo simulation
  - `delta_vega_hedge_mc.py` – Delta-vega hedging using Monte Carlo simulation
  - `delta_vega_hedge_sa.py` – Delta-vega hedging using semi-analytical option pricing
  - `heston_model_mc.py` – Heston model simulation using Monte Carlo
  - `heston_model_sa.py` – Heston model using semi-analytical pricing formula

- `.gitignore` – Standard Python ignore file
- `README.md` – Project overview and instructions



## How to Run

Navigate to the `modeling` directory and execute any Python script:

```bash
cd modeling
python delta_hedge_mc.py
```

Make sure you have the required Python packages installed: numpy, matplotlib, scipy, and pandas.

## License
For academic use only.

