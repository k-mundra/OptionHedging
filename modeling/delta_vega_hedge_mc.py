from modeling.heston_model_mc import sim_heston_paths, heston_greeks, heston_option_price
import numpy as np
import matplotlib.pyplot as plt

 
def dynamic_hedge_vega(hedge_freq=10):
    S0 = 100
    v0 = 0.04
    r = 0.05
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    T = 1.0
    K1 = 110  # prim option strike added
    K2 = 120
    num_steps_outer = 365
    num_paths_outer = 100
 
    # steps
    h = 1e-3     
    h_vega = 1e-4
 
    num_paths_inner = 500
 
    
    hedge_step_days = max(1, int(hedge_freq))
    dt_outer = T / num_steps_outer
    steps_per_day = num_steps_outer / (T * 365)
    hedge_step = int(round(hedge_step_days * steps_per_day))
    hedge_indices = np.arange(0, num_steps_outer + 1, hedge_step)
 
    # outer sims
    np.random.seed(42)
    Z_outer = np.random.randn(num_paths_outer, 2, num_steps_outer)
    S_paths, V_paths = sim_heston_paths(S0, v0, r, kappa, theta, sigma, rho,
                                        dt_outer, num_steps_outer, num_paths_outer, Z_outer)
 
    
    cash = np.zeros(num_paths_outer)  
    stock_pos = np.zeros(num_paths_outer)  
    option2_pos = np.zeros(num_paths_outer)
 
    for t_idx in hedge_indices:
        tau = T - t_idx * dt_outer
 
        if t_idx > 0:
            time_elapsed = hedge_step_days * dt_outer  
            cash *= np.exp(r * time_elapsed)
 
        S_t = S_paths[:, t_idx]
        V_t = V_paths[:, t_idx]
 
        # array initialization
 
        # primary options
        deltas1 = np.zeros_like(S_t)
        vegas1 = np.zeros_like(S_t)
        prices1 = np.zeros_like(S_t)
        
        # hedging options
        deltas2 = np.zeros_like(S_t)
        vegas2 = np.zeros_like(S_t)
        prices2 = np.zeros_like(S_t)  
 
        num_steps_remaining = num_steps_outer - t_idx
 
        for i in range(num_paths_outer):
            Z_inner = np.random.randn(num_paths_inner, 2, num_steps_remaining)

            args = [r, kappa, theta, sigma, rho, dt_outer, num_steps_remaining, num_paths_inner, Z_inner]
            
            # variable stepping
            h = S_t[i] * 1e-2
            h_vega = max(V_t[i] * 1e-2, 1e-4) # set minimum step size to be big enough
            
            s_main, _ = sim_heston_paths(S_t[i], V_t[i], *args)
            s_plus, _ = sim_heston_paths(S_t[i] + h, V_t[i], *args)
            s_minus, _ = sim_heston_paths(S_t[i] - h, V_t[i], *args)
            s_plus_v, _ = sim_heston_paths(S_t[i], V_t[i] + h_vega, *args)
            s_minus_v, _ = sim_heston_paths(S_t[i], V_t[i] - h_vega, *args)
            
            # primary option
            price1 = heston_option_price(s_main, K1, r, tau)
            price_plus1 = heston_option_price(s_plus, K1, r, tau)
            price_minus1 = heston_option_price(s_minus, K1, r, tau)
            price_plus_v1 = heston_option_price(s_plus_v, K1, r, tau)
            price_minus_v1 = heston_option_price(s_minus_v, K1, r, tau)

            delta1 = heston_greeks(price_plus1, price_minus1, h, mode='delta')
            vega1 = heston_greeks(price_plus_v1, price_minus_v1, h_vega, mode='vega')
            
            deltas1[i] = delta1
            vegas1[i] = vega1
            prices1[i] = price1

            # hedging option
            
            price2 = heston_option_price(s_main, K2, r, tau)
            price_plus2 = heston_option_price(s_plus, K2, r, tau)
            price_minus2 = heston_option_price(s_minus, K2, r, tau)
            price_plus_v2 = heston_option_price(s_plus_v, K2, r, tau)
            price_minus_v2 = heston_option_price(s_minus_v, K2, r, tau)
    
            delta2 = heston_greeks(price_plus2, price_minus2, h, mode='delta')
            vega2 = heston_greeks(price_plus_v2, price_minus_v2, h_vega, mode='vega')
            
            deltas2[i] = delta2
            vegas2[i] = vega2
            prices2[i] = price2
        
        # hedge ratios
        
        n2 = (vegas1 * vegas2) / (vegas2**2 + 1e-2) # regularised
        n2 = np.clip(n2, -10, 10) # cap to reasonable values
        n_stock = deltas1 - n2 * deltas2
        
        if t_idx == 0:
            stock_pos = n_stock
            option2_pos = n2
            cash = prices1 - n2 * prices2 - n_stock * S_t
        else:
            stock_diff = n_stock - stock_pos
            cash -= stock_diff * S_t
            stock_pos = n_stock
            
            option2_diff = n2 - option2_pos
            cash -= option2_diff * prices2
            option2_pos = n2
    
    S_T = S_paths[:, -1]
 
    # option payoffs
    payoff1 = np.maximum(S_T - K1, 0) # owed
    payoff2 = np.maximum(S_T - K2, 0) # receive
    
    # final interest
    cash *= np.exp(r * hedge_step_days * dt_outer)
    
    pnl = stock_pos * S_T + option2_pos * payoff2 + cash - payoff1
 
    print(f"Mean PnL: ${np.mean(pnl):.4f}")
    print(f"Std Dev PnL: ${np.std(pnl):.4f}")
 
    plt.figure(figsize=(8, 6))
    plt.hist(pnl, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(np.mean(pnl), color='red', linestyle='--', linewidth=2,
                label=f'Mean = ${np.mean(pnl):.2f}')
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('P&L (dollars $)')
    plt.ylabel('Freq')
    plt.title(f'PnL Distribution, Delta-Vega Hedging (K1=${K1}, K2=${K2})')
    plt.legend()
    plt.grid(True)
    plt.show()
 
    return pnl
 
 
if __name__ == "__main__":
    pnl = dynamic_hedge_vega(hedge_freq=10)