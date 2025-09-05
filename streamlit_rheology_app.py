# Streamlit Rheology Fitting & Plotting App
# Author: ChatGPT (GPT-5 Thinking)
# Description:
#   Upload shear rheology data (shear rate vs shear stress OR viscosity) and
#   fit common flow models (Newtonian, Power-Law, Bingham, Herschel–Bulkley,
#   and Carreau–Yasuda). Visualize fits, residuals, and export parameters
#   and fitted curves. Designed for materials characterization workflows.

import io
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------- Page Config & Style --------------------------- #
st.set_page_config(
    page_title="Rheology – Model Fits",
    page_icon="🌀",
    layout="wide",
)

_DARK_HELP = """
**CSV esperado (exemplos de cabeçalhos):**
- `shear_rate,shear_stress` (s⁻¹, Pa)
- `shear_rate,viscosity` (s⁻¹, Pa·s)

Você pode mapear as colunas manualmente depois do upload. Unidades diferentes
(p.ex. kPa, mPa·s) podem ser convertidas no app.
"""

st.markdown(
    """
    <style>
    .small-note { opacity: 0.7; font-size: 0.9rem; }
    .metric-ok { color: #10B981; }
    .metric-warn { color: #F59E0B; }
    .metric-bad { color: #EF4444; }
    div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ Models ----------------------------------- #
# Stress-based models (tau vs gamma_dot)
def model_newtonian(gamma, eta):
    return eta * gamma

def model_powerlaw(gamma, K, n):
    return K * np.power(gamma, n)

def model_bingham(gamma, tau0, eta_p):
    return tau0 + eta_p * gamma

def model_herschel_bulkley(gamma, tau0, K, n):
    return tau0 + K * np.power(gamma, n)

# Viscosity-based model (eta vs gamma_dot)
def model_carreau_yasuda(gamma, eta0, etainf, lam, a, n):
    # eta(gamma) = eta_inf + (eta0 - eta_inf) * [1 + (lam*gamma)^a]^((n - 1)/a)
    return etainf + (eta0 - etainf) * np.power(1.0 + np.power(lam * gamma, a), (n - 1.0) / a)

@dataclass
class FitResult:
    name: str
    params: Dict[str, float]
    param_errors: Dict[str, float]
    y_pred: np.ndarray
    r2: float
    rmse: float
    aic: float
    bic: float

# --------------------------- Utility Functions --------------------------- #
def goodness_of_fit(y_true: np.ndarray, y_pred: np.ndarray, k_params: int) -> Tuple[float, float, float, float]:
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / max(len(y_true) - k_params, 1)))
    n = len(y_true)
    aic = n * np.log(ss_res / n + 1e-30) + 2 * k_params
    bic = n * np.log(ss_res / n + 1e-30) + k_params * np.log(n)
    return r2, rmse, aic, bic


def safe_curve_fit(func: Callable, x: np.ndarray, y: np.ndarray, p0: List[float], bounds: Tuple[List[float], List[float]]):
    # Robust wrapper: try curve_fit; if it fails, fall back to least_squares with soft_l1 loss
    try:
        popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
        return popt, perr
    except Exception:
        # least_squares expects residual function
        def resid(p):
            return func(x, *p) - y
        lb, ub = bounds
        res = least_squares(resid, x0=p0, bounds=(lb, ub), loss='soft_l1')
        popt = res.x
        # Approximate errors via Jacobian (Gauss–Newton); may be rough
        try:
            _, s, VT = np.linalg.svd(res.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[: s.size]
            cov = VT.T @ (VT / (s ** 2))
            sigma2 = (res.fun @ res.fun) / (len(y) - len(popt))
            pcov = cov * sigma2
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            perr = np.full(len(popt), np.nan)
        return popt, perr


def initial_guess_bounds(model_name: str, x: np.ndarray, y: np.ndarray):
    x = np.asarray(x)
    y = np.asarray(y)
    # Basic statistics
    eta_guess = float(np.median(y / np.clip(x, 1e-12, None)))
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else eta_guess

    if model_name == "Newtonian":
        p0 = [max(eta_guess, 1e-9)]
        lb = [1e-12]
        ub = [1e6]
        names = ["eta"]
    elif model_name == "Power-Law":
        # log-log linearization for guesses
        mask = (x > 0) & (y > 0)
        if mask.sum() >= 2:
            b, a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)  # log(y)=a + b log(x)
            n0 = float(b)
            K0 = float(np.exp(a))
        else:
            n0, K0 = 1.0, max(eta_guess, 1e-9)
        p0 = [max(K0, 1e-12), np.clip(n0, 0.01, 3.0)]
        lb = [1e-12, 0.01]
        ub = [1e6, 3.0]
        names = ["K", "n"]
    elif model_name == "Bingham":
        # linear fit for intercept & slope
        try:
            slope_lin, intercept_lin = np.polyfit(x, y, 1)
            tau0 = float(max(intercept_lin, 0.0))
            eta_p = float(max(slope_lin, 1e-9))
        except Exception:
            tau0, eta_p = 0.0, max(eta_guess, 1e-9)
        p0 = [tau0, eta_p]
        lb = [0.0, 1e-12]
        ub = [1e6, 1e6]
        names = ["tau0", "eta_p"]
    elif model_name == "Herschel–Bulkley":
        # build from Bingham guess + power exponent
        try:
            slope_lin, intercept_lin = np.polyfit(x, y, 1)
            tau0 = float(max(intercept_lin, 0.0))
            n0 = 1.0
            K0 = float(max(slope_lin, 1e-9))
        except Exception:
            tau0, K0, n0 = 0.0, max(eta_guess, 1e-9), 1.0
        p0 = [tau0, K0, n0]
        lb = [0.0, 1e-12, 0.05]
        ub = [1e6, 1e6, 3.0]
        names = ["tau0", "K", "n"]
    elif model_name == "Carreau–Yasuda":
        # viscosity model guesses
        # eta0 ≈ max viscosity at low gamma; etainf ≈ min viscosity at high gamma
        # lambda ~ 1/median(gamma), a ~ 2, n ~ 0.2–1
        p0 = [float(np.nanmax(y)), float(np.nanmin(y)), 1.0 / float(np.nanmedian(x) + 1e-9), 2.0, 0.5]
        lb = [1e-8, 0.0, 1e-6, 0.2, 0.05]
        ub = [1e6, 1e3, 1e6, 5.0, 1.5]
        names = ["eta0", "eta_inf", "lambda", "a", "n"]
    else:
        raise ValueError("Unknown model")

    return p0, (lb, ub), names


def fit_model(model_name: str, x: np.ndarray, y: np.ndarray) -> FitResult:
    if model_name == "Newtonian":
        func = model_newtonian
    elif model_name == "Power-Law":
        func = model_powerlaw
    elif model_name == "Bingham":
        func = model_bingham
    elif model_name == "Herschel–Bulkley":
        func = model_herschel_bulkley
    elif model_name == "Carreau–Yasuda":
        func = model_carreau_yasuda
    else:
        raise ValueError("Model not implemented")

    p0, bounds, names = initial_guess_bounds(model_name, x, y)
    popt, perr = safe_curve_fit(func, x, y, p0, bounds)

    y_pred = func(x, *popt)
    r2, rmse, aic, bic = goodness_of_fit(y, y_pred, len(popt))

    params = {n: float(v) for n, v in zip(names, popt)}
    param_errors = {n: float(e) if np.isfinite(e) else np.nan for n, e in zip(names, perr)}

    return FitResult(name=model_name, params=params, param_errors=param_errors, y_pred=y_pred, r2=r2, rmse=rmse, aic=aic, bic=bic)


def example_dataset(n=50, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gamma = np.logspace(-1, 2, n)  # 0.1 .. 100 s^-1
    # Ground truth: Herschel–Bulkley
    tau0_true, K_true, n_true = 5.0, 0.8, 0.6
    tau = tau0_true + K_true * np.power(gamma, n_true)
    noise = rng.normal(0, 0.05 * np.maximum(tau, 1.0))
    tau_noisy = tau + noise
    df = pd.DataFrame({"shear_rate": gamma, "shear_stress": tau_noisy})
    return df

# ------------------------------ Sidebar ---------------------------------- #
st.sidebar.header("Entrada de dados")
mode = st.sidebar.radio("Tipo de dado carregado:", ["Shear Stress vs Shear Rate", "Viscosity vs Shear Rate"], index=0)

uploaded = st.sidebar.file_uploader("CSV com dados", type=["csv", "txt"])
use_example = st.sidebar.button("Usar dataset de exemplo")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif use_example:
    raw = example_dataset()
else:
    raw = None

if raw is None:
    st.info("Envie um CSV ou clique em *Usar dataset de exemplo*." )
    st.markdown(_DARK_HELP)
    st.stop()

st.sidebar.subheader("Mapeamento de colunas")
cols = list(raw.columns)
col_x = st.sidebar.selectbox("Coluna de shear rate (s⁻¹)", cols, index=0)

if mode.startswith("Shear Stress"):
    col_y = st.sidebar.selectbox("Coluna de shear stress", cols, index=min(1, len(cols)-1))
else:
    col_y = st.sidebar.selectbox("Coluna de viscosidade", cols, index=min(1, len(cols)-1))

st.sidebar.subheader("Unidades & conversões")
# Shear rate
st.sidebar.caption("Shear rate deve estar em s⁻¹")

# Stress or viscosity units
if mode.startswith("Shear Stress"):
    y_unit = st.sidebar.selectbox("Unidade de shear stress", ["Pa", "kPa"], index=0)
else:
    y_unit = st.sidebar.selectbox("Unidade de viscosidade", ["Pa·s", "mPa·s"], index=0)

# Model selection
st.sidebar.header("Modelos para ajustar")
model_choices = ["Newtonian", "Power-Law", "Bingham", "Herschel–Bulkley"]
visc_model = st.sidebar.checkbox("Incluir Carreau–Yasuda (viscosidade)", value=(mode.startswith("Viscosity")))
if visc_model:
    model_choices = model_choices + ["Carreau–Yasuda"]

advanced = st.sidebar.expander("Opções avançadas", expanded=False)
with advanced:
    logx = st.checkbox("Eixo x em log10", value=True)
    logy = st.checkbox("Eixo y em log10 (para tau)", value=False)
    n_points_curve = st.slider("Pontos nas curvas previstas", 50, 2000, 400)
    show_resid = st.checkbox("Mostrar gráfico de resíduos", value=True)
    export_pred = st.checkbox("Exportar curvas previstas", value=True)

# ------------------------------ Data Prep -------------------------------- #
df = raw[[col_x, col_y]].copy()
df.columns = ["shear_rate", "y_in"]

# Drop NA and nonpositive gamma for modeling
before = len(df)
df = df.dropna()
if len(df) < before:
    st.warning(f"Removidas {before - len(df)} linhas com NA.")

# Convert units
if mode.startswith("Shear Stress"):
    if y_unit == "kPa":
        df["y_in"] = df["y_in"].astype(float) * 1e3  # kPa -> Pa
    tau_series = df["y_in"].astype(float).values
    gamma_series = df["shear_rate"].astype(float).values
    # Filtering for nonpositive gamma
    mask = gamma_series > 0
    df = df.loc[mask]
    gamma_series = gamma_series[mask]
    tau_series = tau_series[mask]
else:
    # viscosity data
    if y_unit == "mPa·s":
        df["y_in"] = df["y_in"].astype(float) * 1e-3  # mPa·s -> Pa·s
    eta_series = df["y_in"].astype(float).values
    gamma_series = df["shear_rate"].astype(float).values
    mask = (gamma_series > 0) & (eta_series >= 0)
    df = df.loc[mask]
    gamma_series = gamma_series[mask]
    eta_series = eta_series[mask]
    # For stress-based fits from viscosity, compute tau = eta*gamma
    tau_series = eta_series * gamma_series

# ------------------------------ Fitting ---------------------------------- #
st.header("🌀 Ajuste de modelos reológicos")
st.caption("Carregue os dados, selecione modelos e visualize as curvas com métricas de ajuste.")

fit_results: List[FitResult] = []

for m in model_choices:
    try:
        if m == "Carreau–Yasuda":
            if mode.startswith("Viscosity"):
                y_data = eta_series
            else:
                # Se usuário forneceu stress, convertimos para viscosidade aparente
                eta_from_tau = tau_series / np.clip(gamma_series, 1e-12, None)
                y_data = eta_from_tau
            res = fit_model(m, gamma_series, y_data)
        else:
            # stress-based models
            res = fit_model(m, gamma_series, tau_series)
        fit_results.append(res)
    except Exception as e:
        st.warning(f"Falha ao ajustar {m}: {e}")

# ------------------------------ Plots ------------------------------------ #
# Primary plot: data + fitted curves
x_plot = np.logspace(np.log10(max(np.min(gamma_series), 1e-6)), np.log10(np.max(gamma_series) * 1.05), int(n_points_curve))

fig = go.Figure()

# Scatter data
if mode.startswith("Shear Stress"):
    fig.add_trace(go.Scatter(x=gamma_series, y=tau_series, mode="markers", name="Dados (tau)", marker=dict(size=8)))
else:
    # show two scatters: viscosity and equivalent stress
    fig.add_trace(go.Scatter(x=gamma_series, y=eta_series, mode="markers", name="Dados (eta)", marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=gamma_series, y=tau_series, mode="markers", name="Dados (tau=eta·γ)", marker=dict(size=6, symbol="x")))

# Add fitted curves
for res in fit_results:
    if res.name == "Carreau–Yasuda":
        y_curve = model_carreau_yasuda(x_plot, **res.params)
        fig.add_trace(go.Scatter(x=x_plot, y=y_curve, mode="lines", name=f"{res.name} (η)", line=dict(width=3)))
    else:
        # stress models
        if res.name == "Newtonian":
            y_curve = model_newtonian(x_plot, **res.params)
        elif res.name == "Power-Law":
            y_curve = model_powerlaw(x_plot, **res.params)
        elif res.name == "Bingham":
            y_curve = model_bingham(x_plot, **res.params)
        elif res.name == "Herschel–Bulkley":
            y_curve = model_herschel_bulkley(x_plot, **res.params)
        else:
            continue
        fig.add_trace(go.Scatter(x=x_plot, y=y_curve, mode="lines", name=f"{res.name} (τ)", line=dict(width=3)))

fig.update_layout(
    xaxis_title="Shear rate γ (s⁻¹)",
    yaxis_title="Shear stress τ (Pa) ou Viscosidade η (Pa·s)",
    legend_title="Séries",
    template="plotly_dark",
    height=600,
)

if logx:
    fig.update_xaxes(type="log")
if logy and mode.startswith("Shear Stress"):
    fig.update_yaxes(type="log")

st.plotly_chart(fig, use_container_width=True)

# --------------------------- Fit Summary Table --------------------------- #
if fit_results:
    rows = []
    for r in fit_results:
        row = {
            "Modelo": r.name,
            "R²": r.r2,
            "RMSE": r.rmse,
            "AIC": r.aic,
            "BIC": r.bic,
        }
        for k, v in r.params.items():
            err = r.param_errors.get(k, np.nan)
            row[f"{k}"] = v
            row[f"{k}±"] = err
        rows.append(row)
    tbl = pd.DataFrame(rows)
    st.subheader("Resumo dos ajustes")
    st.dataframe(tbl, use_container_width=True)

    # Downloads
    csv_buf = io.StringIO()
    tbl.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Baixar parâmetros (CSV)", csv_buf.getvalue(), file_name="rheology_fit_parameters.csv", mime="text/csv")

# ----------------------------- Residuals --------------------------------- #
if show_resid and fit_results:
    st.subheader("Resíduos dos ajustes (y - y_pred)")
    tabs = st.tabs([r.name for r in fit_results])
    for tab, r in zip(tabs, fit_results):
        with tab:
            if r.name == "Carreau–Yasuda":
                if mode.startswith("Viscosity"):
                    y_obs = eta_series
                else:
                    y_obs = tau_series / np.clip(gamma_series, 1e-12, None)
            else:
                y_obs = tau_series
            resid = y_obs - r.y_pred
            fig_res = px.scatter(x=gamma_series, y=resid, labels={"x": "γ (s⁻¹)", "y": "Resíduo"})
            fig_res.update_traces(marker=dict(size=8))
            if logx:
                fig_res.update_xaxes(type="log")
            fig_res.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_res, use_container_width=True)

# -------------------------- Export Predicted ----------------------------- #
if export_pred and fit_results:
    all_curves = {"shear_rate": x_plot}
    for r in fit_results:
        if r.name == "Carreau–Yasuda":
            all_curves[f"eta_{r.name}"] = model_carreau_yasuda(x_plot, **r.params)
        else:
            if r.name == "Newtonian":
                y_curve = model_newtonian(x_plot, **r.params)
            elif r.name == "Power-Law":
                y_curve = model_powerlaw(x_plot, **r.params)
            elif r.name == "Bingham":
                y_curve = model_bingham(x_plot, **r.params)
            elif r.name == "Herschel–Bulkley":
                y_curve = model_herschel_bulkley(x_plot, **r.params)
            else:
                continue
            all_curves[f"tau_{r.name}"] = y_curve

    pred_df = pd.DataFrame(all_curves)
    buf = io.StringIO()
    pred_df.to_csv(buf, index=False)
    st.download_button("⬇️ Baixar curvas previstas (CSV)", buf.getvalue(), file_name="rheology_predicted_curves.csv", mime="text/csv")

# --------------------------- Help & Notes -------------------------------- #
with st.expander("Ajuda e Notas"):
    st.markdown(
        """
        **Modelos implementados**
        - *Newtoniano*: τ = η·γ
        - *Ostwald–de Waele (Power-Law)*: τ = K·γⁿ
        - *Bingham plastic*: τ = τ₀ + ηₚ·γ
        - *Herschel–Bulkley*: τ = τ₀ + K·γⁿ
        - *Carreau–Yasuda* (viscosidade): η(γ) = η∞ + (η₀ − η∞)[1 + (λγ)ᵃ]^((n−1)/a)

        **Dicas**
        - Prefira dados com uma boa varredura de γ (pelo menos 1–2 décadas) para
          modelos não-newtonianos e Carreau–Yasuda.
        - Se medir viscosidade, marque *Viscosity vs Shear Rate*. Para comparar
          com modelos de tensão, o app calcula τ = η·γ.
        - Use eixos log para Power-Law/Herschel–Bulkley.
        - Baixe as curvas previstas para superpor em outras figuras.
        
        **Créditos**: SciPy (otimização), Plotly (gráficos), Streamlit (UI).
        """
    )
