def dynamic_hedge(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h, K):
    pass
    '''hedge a portfolio short 1 option and long delta stock, with any cash earning interest at r'''

    '''implement this function by 
    1. running MC sims at every time step
    2. finding the option prices discounted to that time and corresponding greeks
    3. update portfolio
    
    finally return PnL -> should be (M,1) array where M is num of paths'''

# FEEL FREE TO ADD ANY HELPER FUNCTIONS OR PLOTTING CODE AS NECESSARY