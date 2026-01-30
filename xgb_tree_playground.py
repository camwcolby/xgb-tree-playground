import os
import re
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ============================================================
# 0) (Optional) Graphviz PATH injection (kept, but NOT required)
#    We will NOT rely on Graphviz; we fall back to a tree-card view.
# ============================================================
GRAPHVIZ_BIN = r"C:\Users\ccolby\Downloads\Graphviz-14.1.2-win32\bin"
os.environ["PATH"] = GRAPHVIZ_BIN + os.pathsep + os.environ.get("PATH", "")

# Optional sanity check (prints to Streamlit logs)
print("dot found at:", shutil.which("dot"))


# ============================================================
# 1) XGBoost "tree-card" helpers (Graphviz-free)
# ============================================================
_SPLIT_RE = re.compile(r"\[([^\]<>=]+?)<")

def parse_tree_split_features(tree_text: str) -> list[str]:
    """
    Extract split feature names from one tree's text dump.
    Example line: '0:[SCADA:Aer_DO_mgL<1.23] yes=1,no=2,...'
    Returns list of feature names used in splits for that tree.
    """
    return _SPLIT_RE.findall(tree_text)

def top_features_from_trees(tree_texts: list[str], top_n: int = 8) -> list[tuple[str, int]]:
    ctr = Counter()
    for t in tree_texts:
        ctr.update(parse_tree_split_features(t))
    return ctr.most_common(top_n)

def safe_predict_prefix(booster: xgb.Booster, dm: xgb.DMatrix, k: int) -> float:
    """
    Predict using only trees [0..k] if supported; otherwise fallback to full prediction.
    """
    try:
        return float(booster.predict(dm, iteration_range=(0, k + 1))[0])
    except Exception:
        return float(booster.predict(dm)[0])


# ============================================================
# 2) Dummy wastewater dataset (simplified)
# ============================================================
def seasonal_temp(doy: np.ndarray) -> np.ndarray:
    return 14 + 9 * np.sin(2 * np.pi * (doy - 200) / 365.0)

def gen_dummy_wastewater(
    n_days: int = 800,
    seed: int = 42,
    missing_rate: float = 0.06,
    spike_rate: float = 0.015,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    doy = dates.dayofyear.values

    temp_c = seasonal_temp(doy) + rng.normal(0, 0.6, size=n_days)
    flow_mgd = np.clip(
        rng.normal(55, 10, size=n_days) + 10 * np.sin(2 * np.pi * doy / 7),
        15,
        120,
    )
    rain_in = np.clip(rng.gamma(shape=1.3, scale=0.25, size=n_days) - 0.05, 0, 4.0)

    inf_nh4 = np.clip(
        rng.normal(28, 4.5, size=n_days) + 2 * np.sin(2 * np.pi * (doy - 40) / 365),
        10,
        45,
    )
    srt = np.clip(10 + 0.35 * (20 - temp_c) + rng.normal(0, 1.1, size=n_days), 3, 35)
    do = np.clip(
        rng.normal(1.8, 0.45, size=n_days) + 0.15 * np.sin(2 * np.pi * doy / 30),
        0.2,
        4.5,
    )
    ferric = np.clip(
        rng.normal(6.0, 2.0, size=n_days) + 0.7 * rng.normal(0, 1, size=n_days),
        0,
        25,
    )

    # Hidden nonlinear "risk" logic (dummy)
    nitrif_stress = (
        0.9 * (temp_c > 18).astype(float) * (srt < 8).astype(float)
        + 0.6 * (do < 1.2).astype(float)
        + 0.3 * (inf_nh4 > 32).astype(float)
    )
    solids_stress = 0.5 * (flow_mgd > 70).astype(float) + 0.7 * (rain_in > 0.6).astype(float)
    chem_help = -0.35 * np.tanh((ferric - 6) / 6)

    score = (
        -2.0
        + 0.02 * (flow_mgd - 55)
        + 0.55 * solids_stress
        + 0.80 * nitrif_stress
        + 0.18 * np.tanh((inf_nh4 - 28) / 6)
        + 0.12 * (rain_in > 1.0).astype(float)
        + chem_help
        + rng.normal(0, 0.25, size=n_days)
    )
    p_bad_today = 1 / (1 + np.exp(-score))
    bad_today = (rng.random(n_days) < p_bad_today).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "WIMS:Inf_Flow_MGD": flow_mgd,
            "SCADA:Rain_in": rain_in,
            "WIMS:Inf_NH4_mgL": inf_nh4,
            "SCADA:Aer_Temp_C": temp_c,
            "SCADA:SRT_days": srt,
            "SCADA:Aer_DO_mgL": do,
            "SCADA:Ferric_mgL": ferric,
            "event_today": bad_today,
        }
    )

    # Missingness
    feat_cols = [c for c in df.columns if c not in ["date", "event_today"]]
    for c in feat_cols:
        m = rng.random(n_days) < missing_rate
        df.loc[m, c] = np.nan

    # Spikes
    for c in ["WIMS:Inf_Flow_MGD", "SCADA:Aer_DO_mgL", "SCADA:SRT_days"]:
        m = rng.random(n_days) < spike_rate
        df.loc[m, c] = df.loc[m, c] + rng.normal(0, 20, size=m.sum())

    return df

def future_label(df: pd.DataFrame, lead_days: int = 3) -> pd.DataFrame:
    """label=1 if any event occurs in t+1..t+lead_days"""
    e = df["event_today"].values.astype(int)
    y = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        j1 = i + 1
        j2 = min(i + lead_days, len(df) - 1)
        if j1 <= j2 and e[j1 : j2 + 1].max() == 1:
            y[i] = 1
    out = df.copy()
    out["label_future_noncompliance"] = y
    return out.iloc[:-lead_days].copy()

def add_features(
    df: pd.DataFrame,
    use_rolling: bool,
    use_lags: bool,
    add_season: bool,
    roll_days: int = 7,
    lag_days: int = 3,
) -> pd.DataFrame:
    out = df.copy()
    base_feats = [c for c in out.columns if c not in ["date", "event_today", "label_future_noncompliance"]]

    if add_season:
        doy = out["date"].dt.dayofyear.astype(float)
        out["season_sin"] = np.sin(2 * np.pi * doy / 365.0)
        out["season_cos"] = np.cos(2 * np.pi * doy / 365.0)

    if use_rolling:
        for c in base_feats:
            out[f"{c}_roll{roll_days}"] = out[c].rolling(
                roll_days, min_periods=max(2, roll_days // 2)
            ).mean()

    if use_lags:
        for k in range(1, lag_days + 1):
            for c in base_feats:
                out[f"{c}_lag{k}"] = out[c].shift(k)

    return out

def clip_and_impute(
    X: pd.DataFrame, clip_outliers: bool = True, loq: float = 0.01, hiq: float = 0.99
) -> pd.DataFrame:
    X2 = X.copy()
    for c in X2.columns:
        X2[c] = pd.to_numeric(X2[c], errors="coerce")

    if clip_outliers:
        for c in X2.columns:
            lo = np.nanquantile(X2[c].values, loq)
            hi = np.nanquantile(X2[c].values, hiq)
            X2[c] = X2[c].clip(lo, hi)

    med = X2.median(numeric_only=True)
    X2 = X2.fillna(med)
    return X2


# ============================================================
# 3) Streamlit UI (simple)
# ============================================================
st.set_page_config(page_title="XGBoost Wastewater Playground", layout="wide")
st.title("XGBoost Wastewater Playground (Simple)")
st.caption("Toggle levers, click Run, watch trees boost one-by-one, then see Compliance Risk + Accuracy.")

cL, cR = st.columns([1, 2], gap="large")

with cL:
    st.subheader("Goal")
    st.write("Predict **non-compliance within the next N days** (dummy data).")

    lead_days = st.slider("Forecast window (days ahead)", 1, 14, 3, 1)

    st.subheader("Levers (refinements)")
    use_lags = st.checkbox("Lag features (process delay)", value=True)
    use_rolling = st.checkbox("Rolling mean (denoise)", value=True)
    add_season = st.checkbox("Seasonality features", value=True)
    clip_outliers = st.checkbox("Clip outliers (sensor spikes)", value=True)

    st.subheader("Model size")
    n_estimators = st.slider("Number of trees", 10, 200, 60, 5)
    max_depth = st.slider("Tree depth", 1, 6, 3, 1)
    learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.08, 0.01)

    st.subheader("Alert threshold")
    risk_threshold = st.slider("Flag non-compliance if risk >", 0.05, 0.80, 0.20, 0.01)

    st.subheader("Scenario to score (pretend this is today's plant state)")
    flow_s = st.slider("Inf Flow (MGD)", 0.0, 120.0, 55.0, 1.0)
    do_s = st.slider("Aer DO (mg/L)", 0.0, 6.0, 1.8, 0.1)
    srt_s = st.slider("SRT (days)", 0.0, 35.0, 12.0, 0.5)
    temp_s = st.slider("Aer Temp (°C)", 0.0, 30.0, 18.0, 0.5)
    nh4_s = st.slider("Inf NH4 (mg/L)", 0.0, 45.0, 28.0, 0.5)
    rain_s = st.slider("Rain (in)", 0.0, 4.0, 0.0, 0.1)
    ferric_s = st.slider("Ferric (mg/L)", 0.0, 25.0, 6.0, 0.5)

    run_btn = st.button("Run (Train + Animate Trees)", type="primary")


# ============================================================
# 4) Train + Animate
# ============================================================
def safe_auc(y_true: pd.Series, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, p))
    except Exception:
        return float("nan")

def evaluate(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    risk_threshold: float,
) -> dict:
    p_tr = model.predict_proba(X_train)[:, 1]
    p_te = model.predict_proba(X_test)[:, 1]
    return {
        "train_auc": safe_auc(y_train, p_tr),
        "test_auc": safe_auc(y_test, p_te),
        "train_ll": log_loss(y_train, p_tr),
        "test_ll": log_loss(y_test, p_te),
        "test_acc@thr": accuracy_score(y_test, (p_te >= risk_threshold).astype(int)),
    }


if run_btn:
    with cR:
        st.subheader("Boosting animation (Graphviz-free)")
        status = st.empty()
        prog = st.progress(0)
        tree_slot = st.empty()
        risk_slot = st.empty()
        risk_chart = st.empty()

        status.info("Generating dummy facility data…")
        df = gen_dummy_wastewater(n_days=900, seed=42, missing_rate=0.08, spike_rate=0.02)
        df = future_label(df, lead_days=lead_days)
        df = add_features(
            df,
            use_rolling=use_rolling,
            use_lags=use_lags,
            add_season=add_season,
            roll_days=7,
            lag_days=3,
        )

        df = df.dropna(subset=["label_future_noncompliance"]).copy()
        y = df["label_future_noncompliance"].astype(int)

        X = df.drop(columns=["date", "event_today", "label_future_noncompliance"])
        X = clip_and_impute(X, clip_outliers=clip_outliers, loq=0.01, hiq=0.99)

        # time split (chronological)
        split_idx = int(0.75 * len(X))
        X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
        y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

        # imbalance weight
        pos = max(int(y_train.sum()), 1)
        neg = max(int((1 - y_train).sum()), 1)
        spw = neg / pos

        status.info("Training XGBoost…")
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=0,
            scale_pos_weight=spw,
        )
        model.fit(X_train, y_train)

        # Scenario row (demo)
        scenario = pd.DataFrame(
            [
                {
                    "WIMS:Inf_Flow_MGD": flow_s,
                    "SCADA:Rain_in": rain_s,
                    "WIMS:Inf_NH4_mgL": nh4_s,
                    "SCADA:Aer_Temp_C": temp_s,
                    "SCADA:SRT_days": srt_s,
                    "SCADA:Aer_DO_mgL": do_s,
                    "SCADA:Ferric_mgL": ferric_s,
                }
            ]
        )

        # Match training features: fill engineered columns with medians (demo shortcut)
        med = X_train.median(numeric_only=True)
        for c in X_train.columns:
            if c not in scenario.columns:
                scenario[c] = med[c]
        scenario = scenario[X_train.columns]
        scenario = clip_and_impute(scenario, clip_outliers=False)

        booster = model.get_booster()
        metrics = evaluate(model, X_train, y_train, X_test, y_test, risk_threshold)

        # Animation prep
        dm = xgb.DMatrix(scenario.values, feature_names=list(scenario.columns))
        all_tree_texts = booster.get_dump(with_stats=True)

        status.success("Animating boosting cycle (tree-by-tree)…")
        steps = min(int(n_estimators), 40)

        risk_hist: list[float] = []

        for k in range(steps):
            prog.progress(int((k + 1) / steps * 100))

            # risk with trees 0..k
            p_k = safe_predict_prefix(booster, dm, k)
            risk_hist.append(p_k)

            # update convergence chart
            risk_df = pd.DataFrame(
                {"Tree": np.arange(1, len(risk_hist) + 1), "Risk": risk_hist}
            ).set_index("Tree")
            risk_chart.line_chart(risk_df)

            # ---- Tree Card (Graphviz-free) ----
            tree_text = all_tree_texts[k]
            tree_feats = parse_tree_split_features(tree_text)

            tree_slot.markdown(f"### Tree {k + 1} / {steps}")

            if len(tree_feats) == 0:
                tree_slot.write("No splits detected in this tree (rare).")
            else:
                seen = set()
                uniq = []
                for f in tree_feats:
                    if f not in seen:
                        uniq.append(f)
                        seen.add(f)

                tree_slot.write("**Splits used by this tree:**")
                tree_slot.code("\n".join(uniq[:12]), language="text")

            tops = top_features_from_trees(all_tree_texts[: k + 1], top_n=8)
            tree_slot.write("**Top split features so far (trees 1 → current):**")
            if len(tops) > 0:
                denom = max(1, tops[0][1])
                for feat, cnt in tops:
                    tree_slot.progress(min(cnt / denom, 1.0), text=f"{feat}  (splits: {cnt})")
            else:
                tree_slot.write("(No split features detected yet.)")

            label = "NON-COMPLIANT risk" if p_k >= risk_threshold else "COMPLIANT (low risk)"
            risk_slot.markdown(
                f"""
**Tree {k + 1} / {steps}**  
Predicted risk (next **{lead_days}d**): **{p_k:.1%}**  
Status at threshold {risk_threshold:.0%}: **{label}**
"""
            )

            time.sleep(0.08)

        prog.progress(100)

        st.divider()
        st.subheader("Final result (after all trees)")

        p_final = float(model.predict_proba(scenario)[:, 1][0])
        final_label = "NON-COMPLIANT" if p_final >= risk_threshold else "COMPLIANT"

        st.markdown(
            f"""
### Prediction
- **Risk (next {lead_days} days):** **{p_final:.1%}**
- **Classification @ {risk_threshold:.0%}:** **{final_label}**
"""
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test ROC-AUC", "—" if np.isnan(metrics["test_auc"]) else f"{metrics['test_auc']:.3f}")
        c2.metric("Test logloss", f"{metrics['test_ll']:.3f}")
        c3.metric(f"Test accuracy @ {risk_threshold:.0%}", f"{metrics['test_acc@thr']:.3f}")
        c4.metric("Train ROC-AUC", "—" if np.isnan(metrics["train_auc"]) else f"{metrics['train_auc']:.3f}")

        st.markdown("### How to interpret this")
        st.markdown(
            f"""
**What the animation means**
- Each frame represents **one tree** added to the boosted model.
- XGBoost builds a strong model by adding many small trees; each one makes a **correction** to the risk estimate.

**What the tree card means**
- “Splits used by this tree” lists which WIMS/SCADA tags the tree used to make decisions.
- “Top split features so far” shows which tags the model is leaning on most as it learns.

**What “Compliant vs Non-compliant” means**
- The model outputs a probability of non-compliance in the next **{lead_days} days**.
- You convert that probability into an alert using threshold **{risk_threshold:.0%}**.
  - Risk below threshold: **Compliant (low risk)**
  - Risk above threshold: **Non-compliant (high risk)**

**How the levers help**
- **Lags**: captures delay effects (hydraulics, biology, solids inventory).
- **Rolling mean**: reduces SCADA noise; highlights sustained process state.
- **Seasonality**: captures temperature-driven regime shifts.
- **Outlier clipping**: reduces impact of sensor spikes.

**Accuracy**
- “Test” metrics reflect performance on unseen (later) data.
"""
        )
else:
    with cR:
        st.info("Set the levers and click **Run (Train + Animate Trees)** to see boosting animate.")
