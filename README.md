# HedgeStorm: Quantitative Risk Management During Market Shocks with Delta and Vega Hedging in a Heston-Based Framework

Modeling & Simulation Course Project

### Group 19 - Kashvi Mundra, Aishwarya Pendyala, Anusha Saha, Soumil Sahu

[Link to overleaf doc (READ ONLY)](https://www.overleaf.com/read/xhndsyqxvgyj#ab3d43)

---

A Python-based simulation framework for evaluating hedging strategies under stochastic volatility, particularly during market shocks such as COVID-19.

Overview

This project models option pricing using the Heston stochastic volatility model, and evaluates delta and delta-vega hedging strategies. We explore hedging effectiveness under both normal and volatile market conditions, using historical Apple options data.

Key components:

Model calibration using a semi-analytical Heston solution

Monte Carlo simulations to estimate PnL for hedged portfolios

Dynamic hedging strategies with sensitivity to both price (delta) and volatility (vega)

Structure
OptionHedging/
├── datasets/
│   └── AAPL_option_data.csv  # Apple options data
├── modeling/
│   ├── calibration.py
│   ├── delta_hedge_mc.py
│   ├── delta_vega_hedge_mc.py
│   ├── delta_vega_hedge_sa.py
│   ├── heston_model_mc.py
│   └── heston_model_sa.py
├── .gitignore
└── README.md

Running the Code

Navigate into the modeling directory and run any script:

cd modeling
python heston_model_sa.py


Each script simulates different aspects of the hedging framework. Ensure all dependencies (NumPy, SciPy, Matplotlib, Pandas) are installed.
