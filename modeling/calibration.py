import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime as dt
from datetime import date
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

def heston_cf(u, spot, v_init, mean_rev, v_long, vol_vol, corr, var_rp, ttm, rate):
    drift_adj = mean_rev + var_rp
    scale = mean_rev * v_long
    psi = corr * vol_vol * u * 1j
    discriminant = np.sqrt((psi - drift_adj)**2 + vol_vol**2 * (u**2 + 1j*u))
    G = (drift_adj - psi + discriminant) / (drift_adj - psi - discriminant)
    term1 = np.exp(rate * 1j * u * ttm)
    term2 = spot ** (1j * u) * ((1 - G*np.exp(discriminant * ttm)) / (1 - G)) ** (-2 * scale / vol_vol**2)
    term3 = np.exp((scale * (drift_adj - psi + discriminant) * ttm) / vol_vol**2 +
                   v_init * (drift_adj - psi + discriminant) *
                   (1 - np.exp(discriminant * ttm)) /
                   (vol_vol**2 * (1 - G*np.exp(discriminant * ttm))))
    return term1 * term2 * term3

def _heston_integrand(u, spot, strike, v_init, mean_rev, v_long, vol_vol, corr, var_rp, ttm, rate):
    cf_args = (spot, v_init, mean_rev, v_long, vol_vol, corr, var_rp, ttm, rate)
    top = np.exp(rate * ttm) * heston_cf(u - 1j, *cf_args) - strike * heston_cf(u, *cf_args)
    bot = 1j * u * (strike ** (1j * u))
    return top / bot


def price_heston_rect(spot, strike, v0, kappa, theta, sigma, rho, lam, ttm, rate):
    upper = 100
    steps = 10_000
    du = upper / steps
    args_cf = (spot, v0, kappa, theta, sigma, rho, lam, ttm, rate)
    acc = 0.0
    for i in range(1, steps):
        u = du * (2*i + 1) * 0.5
        numerator = np.exp(rate * ttm)*heston_cf(u - 1j, *args_cf) - strike * heston_cf(u, *args_cf)
        denominator = 1j * u * strike**(1j * u)
        acc += du * numerator / denominator
    return np.real((spot - strike*np.exp(-rate*ttm))/2 + acc/np.pi)

maturity_grid = np.array([1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yield_values = (np.array([4.17,4.42,4.53,4.70,4.77,4.72,4.40,4.18,3.94,3.89,3.79,4.06, 3.88]) / 100).astype(float)
yield_curve, _ = calibrate_nss_ols(maturity_grid, yield_values)

opt_data = pd.read_csv("aapl_options.csv", parse_dates=["date", "expiration"])
opt_data.columns = opt_data.columns.str.lower().str.strip()
opt_data["mid_price"] = (opt_data["ask"] + opt_data["bid"]) * 0.5

stk = pd.read_csv("appl_stock.csv")
stk["date"] = pd.to_datetime(stk["Date"])
stk = stk.rename(columns={"Price": "spot_px"})[["date", "spot_px"]]

opt_data = pd.merge(opt_data, stk, on="date", how="left")
unique_days = opt_data["date"].unique()

def run_heston_calibration(data, day, option_kind="Call", show=True):
    subset = data[(data["date"] == day) & (data["call_put"] == option_kind)]
    S_ref = subset["spot_px"].iloc[0]
    grouped = (
        subset.groupby("expiration")
              .apply(lambda g: {"strks": g["strike"].tolist(),
                                "mkt": g["mid_price"].tolist()})
              .to_dict()
    )
    strike_sets = [v["strks"] for v in grouped.values()]
    valid_strikes = sorted(set.intersection(*map(set, strike_sets)))
    maturities, mkt_rows = [], []
    ref_day = subset["date"].iloc[0]
    for exp_dt, quotes in grouped.items():
        tau = (exp_dt - ref_day).days / 365.25
        maturities.append(tau)
        mkt_rows.append([quotes["mkt"][i] 
                         for i, s in enumerate(quotes["strks"]) if s in valid_strikes])
    arr = np.array(mkt_rows, dtype=object)
    df_surf = pd.DataFrame(arr, index=maturities, columns=valid_strikes)
    melted = df_surf.melt(ignore_index=False).reset_index()
    melted.columns = ["tau", "strike", "mkt_px"]
    melted["rate"] = melted["tau"].apply(yield_curve)
    τ = melted["tau"].to_numpy(float)
    K = melted["strike"].to_numpy(float)
    r = melted["rate"].to_numpy(float)
    M = melted["mkt_px"].to_numpy(float)
    def objective(p):
        v0, κ, θ, η, ρ, λ = p
        val = price_heston_rect(S_ref, K, v0, κ, θ, η, ρ, λ, τ, r)
        return np.mean((M - val)**2)
    bounds = [(1e-3,0.9),(1e-3,5),(1e-3,0.1),(1e-2,1),(-1,0),(-1,1)]
    start = [0.1,3,0.05,0.3,-0.8,0.03]
    sol = minimize(objective, start, method="SLSQP",
                   bounds=bounds, tol=1e-3, options={"maxiter":10000})
    fit_vals = price_heston_rect(S_ref, K, *sol.x, τ, r)
    melted["model_px"] = fit_vals
    if show:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_mesh3d(x=melted.tau, y=melted.strike, z=melted.model_px,
                       opacity=0.55, color="royalblue")
        fig.add_scatter3d(x=melted.tau, y=melted.strike, z=melted.mkt_px,
                          mode="markers")
        fig.update_layout(
            title=f"Heston Calibration {day.date()}",
            scene=dict(
                xaxis_title="Maturity",
                yaxis_title="Strike",
                zaxis_title="Price"
            ),
            width=800,
            height=800
        )
        fig.show()
    return sol

all_results = {}
for dt_val in unique_days:
    print("Calibrating:", dt_val)
    res = run_heston_calibration(opt_data, dt_val)
    all_results[dt_val] = res
    print("Params:", res.x)
