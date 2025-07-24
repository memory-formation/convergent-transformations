
import numpy as np
from scipy.optimize import curve_fit
from numpy.random import choice

# Power-law model
def power_law(x, a, b, c):
    return a * x**b + c

# Power-law model
def power_law_zero(x, a, b):
    return a * x**b

# Function to fit, bootstrap CI, and compute pseudo-R²
def fit_powerlaw_with_bootstrap(x, y, n_boot=1000, p0=[1, 0.5, 0], n_points=200):
    # Ensure numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Fit original model
    params, _ = curve_fit(power_law, x, y, p0=p0, maxfev=5000)
    y_pred = power_law(x, *params)
    #print(params)

    # Compute pseudo-R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot

    # Define x values for smooth curve
    x_fit = np.linspace(min(x), max(x), n_points)
    y_fit = power_law(x_fit, *params)

    # Bootstrap confidence intervals
    bootstrap_preds = []
    for _ in range(n_boot):
        indices = choice(range(len(x)), size=len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        try:
            popt, _ = curve_fit(power_law, x_sample, y_sample, p0=p0, maxfev=5000)
            y_boot = power_law(x_fit, *popt)
            bootstrap_preds.append(y_boot)
        except RuntimeError:
            continue  # skip failed fits

    bootstrap_preds = np.array(bootstrap_preds)
    y_lower, y_upper = np.percentile(bootstrap_preds, [2.5, 97.5], axis=0)

    return x_fit, y_fit, y_lower, y_upper, r_squared

# Adjusted function for fitting with bootstrap and pseudo-R²
def fit_powerlaw_zero_with_bootstrap(x, y, n_boot=1000, p0=[1, 0.5], n_points=200):
    x = np.asarray(x)
    y = np.asarray(y)

    # Fit the simplified model
    params, _ = curve_fit(power_law_zero, x, y, p0=p0, maxfev=5000)
    y_pred = power_law_zero(x, *params)

    # Compute pseudo-R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot

    # Generate fitted curve
    x_fit = np.linspace(min(x), max(x), n_points)
    y_fit = power_law_zero(x_fit, *params)

    # Bootstrap confidence intervals
    bootstrap_preds = []
    for _ in range(n_boot):
        indices = choice(range(len(x)), size=len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        try:
            popt, _ = curve_fit(power_law_zero, x_sample, y_sample, p0=p0, maxfev=5000)
            y_boot = power_law_zero(x_fit, *popt)
            bootstrap_preds.append(y_boot)
        except RuntimeError:
            continue

    bootstrap_preds = np.array(bootstrap_preds)
    y_lower, y_upper = np.percentile(bootstrap_preds, [2.5, 97.5], axis=0)

    return x_fit, y_fit, y_lower, y_upper, r_squared, params