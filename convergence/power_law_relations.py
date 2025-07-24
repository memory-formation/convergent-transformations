
import numpy as np
import seaborn as sns
import pandas as pd
from .power_law import fit_powerlaw_zero_with_bootstrap
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import trange

def plot_vars(df, filename_hcp, df_params=None, figsize=None, names=None, **kwargs):
    columns = df.columns#[1:]
    n_cols = len(columns)-1
    figsize = figsize or (n_cols * 4, n_cols * 4)
    fig, axes = plt.subplots(nrows=n_cols, ncols=n_cols, figsize=figsize, sharex='col', sharey='row', **kwargs)
    hcp = pd.read_csv(filename_hcp)
    colors = hcp.query("roi<=180").area_color.tolist()
    for j, col1 in enumerate(columns):
        
        for i, col2 in enumerate(columns):
            
            if i<j and i != j-1:
                axes[i, j-1].axis('off')
            if i <= j:
                continue

            
            ax = axes[i-1, j]
            #sns.scatterplot(x=df[col1], y=df[col2], ax=ax, c=colors)
            sns.scatterplot(x=df[col1], y=df[col2], ax=ax, c="lightgray")
            x_fit, y_fit, y_lower, y_upper, r_squared, params = fit_powerlaw_zero_with_bootstrap(
                df[col1], df[col2])
            r2 = f"{r_squared:.2f}" #if r_squared<0.99 else f"{r_squared:.3f}"
            ax.plot(x_fit, y_fit, color='maroon', label=f'$y = {params[0]:.2f}x^{{{params[1]:.2f}}}$') # ($R^2={r2}$)
            ax.fill_between(x_fit, y_lower, y_upper, color='maroon', alpha=0.2)

            if df_params is not None:
                a1, b1 = df_params.loc[col1].a, df_params.loc[col1].b
                a2, b2 = df_params.loc[col2].a, df_params.loc[col2].b
                C = a2/(a1**(b2/b1))
                B = b2/b1
                # Get lims from x_fit
                x_min, x_max = np.min(x_fit), np.max(x_fit)
                
                l = np.linspace(0, x_max, 100)
                ax.plot(l, C * l**B, color='black', linestyle='--', 
                        label=f'$y = {C:.2f}x^{{{B:.2f}}}$')


            #ax.set_title(f'{col1} vs {col2} (R²={r_squared:.2f})')
            ax.legend(fontsize=9)
            sns.despine(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if names is not None:
                if j == 0:
                    ax.set_ylabel(names[col2])
                if i == n_cols:
                    ax.set_xlabel(names[col1])
                if i-1 == j:
                    #ax.set_title(names[col1])
                    ax.set_title("")
                if i == j:
                    #ax.set_title(names[col2])
                    ax.set_title("")
    return fig, axes



def fit_power_ceiling(df, measures=None, *, intercept=False,
                      n_restart=10, seed=0, verbose=False,
                      a_bounds=(1e-8, 1.0), b_bounds=(1e-8, 1.0),
                      d_bounds=(-0.05, 0.05)):
    """
    General N-measure fit:   X_ij ≈ d_j + a_j * r_i ** b_j

    Parameters
    ----------
    intercept : bool
        If True, fit a per-measure intercept d_j (initialised to 0).
        If False, the model reverts to  a_j * r_i ** b_j .
    d_bounds  : tuple (low, high)
        Bounds for each intercept when intercept=True.

    Returns
    -------
    pars_df : DataFrame ['measure','d','a','b','R2_col']  (d column NaN if intercept=False)
    pred_df : DataFrame rows=parcels, cols=measures plus r_hat / diagnostics.
    """
    rng = np.random.default_rng(seed)

    if measures is None:
        measures = df.columns.tolist()
    X = df[measures].values.astype(float)
    n_roi, n_m = X.shape
    roi_idx = df.index

    # ---------- helper pack / unpack ---------------------------------------
    def unpack(p):
        r = p[:n_roi]
        a = p[n_roi:n_roi + n_m]
        b = p[n_roi + n_m:n_roi + 2 * n_m]
        if intercept:
            d = p[-n_m:]
        else:
            d = np.zeros(n_m)
        return r, a, b, d

    def mse(p):
        r, a, b, d = unpack(p)
        pred = d[None, :] + (r[:, None] ** b[None, :]) * a[None, :]
        return np.mean((pred - X) ** 2)

    # ---------- initial guess ----------------------------------------------
    r0 = np.maximum(X.mean(1), 1e-4)
    a0 = np.maximum(X.mean(0), 1e-4)
    b0 = np.ones(n_m)
    d0 = np.zeros(n_m) if intercept else np.array([])

    base_p0 = np.concatenate([r0, a0, b0, d0])

    best = None
    estimated_r_hats = []
    for k in trange(n_restart, leave=False):
        # jitter
        r_init = r0 * rng.lognormal(0, 0.25, n_roi)
        a_init = a0 * rng.lognormal(0, 0.25, n_m)
        b_init = b0 * rng.lognormal(0, 0.25, n_m)
        d_init = rng.uniform(*d_bounds, n_m) if intercept else np.array([])

        p0 = np.concatenate([r_init, a_init, b_init, d_init])

        # bounds
        bounds = (
            [(1e-8, 1)] * n_roi +          # r_i
            [a_bounds] * n_m +                # a_j
            [b_bounds] * n_m +                # b_j
            ([d_bounds] * n_m if intercept else [])  # d_j
        )

        res = minimize(mse, p0, method="L-BFGS-B", bounds=bounds)
        if verbose:
            print(f"restart {k}: mse={res.fun:.4e}")

        # Save the estimated r_hat for diagnostics

        # Check if res converged
        if res.success:
            r_hat, _, _, _ = unpack(res.x)
            estimated_r_hats.append(r_hat)

        if best is None or res.fun < best.fun:
            best = res
        

    if verbose:
        print("Best L-BFGS-B msg:", best.message, "final mse=", best.fun)

    # ---------- results & diagnostics --------------------------------------
    r_hat, a_hat, b_hat, d_hat = unpack(best.x)
    pred = d_hat[None, :] + (r_hat[:, None] ** b_hat[None, :]) * a_hat[None, :]
    resid = X - pred

    ss_tot_c = ((X - X.mean(0)) ** 2).sum(0)
    R2_col = 1 - (resid**2).sum(0) / ss_tot_c

    pars_df = pd.DataFrame({
        'measure': measures,
        'd': d_hat if intercept else np.nan,
        'a': a_hat,
        'b': b_hat,
        'R2_col': R2_col
    })

    mae_row = np.abs(resid).mean(1)
    ss_tot_r = ((X - X.mean(1, keepdims=True)) ** 2).sum(1)
    R2_row = 1 - (resid**2).sum(1) / ss_tot_r
    R2_row[np.isnan(R2_row)] = 0.0

    pred_df = pd.DataFrame(pred, index=roi_idx, columns=measures)
    pred_df.insert(0, 'r_hat', r_hat)
    pred_df.insert(1, 'MAE_parcel', mae_row)
    pred_df.insert(2, 'R2_parcel', R2_row)

    return pars_df, pred_df, np.array(estimated_r_hats)

