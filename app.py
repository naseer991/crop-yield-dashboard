import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Page config

st.set_page_config(
    page_title="Crop Yield Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TARGET = "hg/ha_yield"
RAIN, PEST, TEMP = "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
NUM_COLS = ["Year", RAIN, PEST, TEMP]
CAT_COLS = ["Area", "Item"]
FEATURES = NUM_COLS + CAT_COLS
PX_TEMPLATE = "plotly_white"
# Works on both old (use_container_width) and new (width="stretch") Streamlit
STRETCH = ({"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 49)
           else {"use_container_width": True})
COLOR_SEQ = px.colors.qualitative.Safe

# Clean dictionary for readable labels
LABELS = {
    TARGET: "Yield (hg/ha)",
    RAIN: "Avg Rainfall (mm/year)",
    PEST: "Pesticides (tonnes)",
    TEMP: "Avg Temp (°C)",
    "Year": "Year",
    "Item": "Crop",
    "Area": "Country"
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%;}
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
      div[data-testid="stMetric"] {
          background: rgba(46,125,50,0.08); border: 1px solid rgba(46,125,50,0.25);
          padding: 10px 14px; border-radius: 10px;}
      .stTabs [data-baseweb="tab-list"] {gap: 6px;}
      .stTabs [data-baseweb="tab"] {padding: 8px 18px; border-radius: 8px 8px 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig, h=380):
    fig.update_layout(
        template=PX_TEMPLATE, height=h, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
    )
    return fig



# Data loading + cleaning

@st.cache_data(show_spinner="Cleaning data...")
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df[RAIN] = pd.to_numeric(df[RAIN], errors="coerce")
    df = df.dropna(subset=[RAIN]).reset_index(drop=True)
    return df


@st.cache_data
def read_csv(path_or_buf):
    return pd.read_csv(path_or_buf)


def load_data():
    for p in ("data/yield_df.csv", "yield_df.csv"):
        if os.path.exists(p):
            return clean(read_csv(p))
    st.title("🌾 Crop Yield Dashboard")
    st.info("`yield_df.csv` was not found in `./data/` or next to `app.py`. Upload it below.")
    up = st.file_uploader("Upload yield_df.csv", type="csv")
    if up is None:
        st.stop()
    df = clean(read_csv(up))
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    return df


df_all = load_data()


# Model training (cached)
# --------------------------------------------------------------------------
def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


@st.cache_resource(show_spinner="Training Random Forest...")
def train_final(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200, max_features=0.5, min_samples_leaf=1,
            random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {
        "R2": r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
    }
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    combined = {}
    for n, v in imp.items():
        orig = n.split("__", 1)[1]
        for c in CAT_COLS:
            if orig.startswith(c + "_"):
                orig = c
        combined[orig] = combined.get(orig, 0) + v
    return pipe, metrics, y_te.reset_index(drop=True), pd.Series(pred), imp, pd.Series(combined)


@st.cache_resource(show_spinner="Comparing models (this can take a minute)...")
def compare_models(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    prep = make_preprocessor()
    Xtr, Xte = prep.fit_transform(X_tr), prep.transform(X_te)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "KNN": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1)
    except ImportError:
        pass
    rows = []
    for name, m in models.items():
        m.fit(Xtr, y_tr)
        p = m.predict(Xte)
        rows.append({
            "Model": name,
            "Train R2": r2_score(y_tr, m.predict(Xtr)),
            "Test R2": r2_score(y_te, p),
            "MAE": mean_absolute_error(y_te, p),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, p))),
        })
    return pd.DataFrame(rows).sort_values("Test R2", ascending=False).reset_index(drop=True)



# Header

st.title("🌾 Crop Yield Dashboard")
st.caption("Exploratory analysis and machine-learning prediction of crop yield (hg/ha)")

# Horizontal filter bar (applies to EDA tabs)
with st.container(border=True):
    f1, f2, f3 = st.columns([2, 2, 2])
    crops_all = sorted(df_all["Item"].unique())
    areas_all = sorted(df_all["Area"].unique())
    ymin, ymax = int(df_all["Year"].min()), int(df_all["Year"].max())
    
    # CHANGED: Default to all crops and areas so it doesn't look broken on load
    sel_crops = f1.multiselect("Crops", crops_all, default=crops_all)
    sel_areas = f2.multiselect("Countries (empty = all)", areas_all, default=areas_all)
    sel_years = f3.slider("Year range", ymin, ymax, (ymin, ymax))

df = df_all[
    df_all["Item"].isin(sel_crops)
    & df_all["Year"].between(*sel_years)
    & (df_all["Area"].isin(sel_areas) if sel_areas else True)
]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

tab_over, tab_eda, tab_geo, tab_model, tab_pred = st.tabs(
    ["📋 Overview", "📊 EDA", "🌍 Countries & Trends", "🤖 Model Performance", "🎯 Predict Yield"]
)


# Overview

with tab_over:
    k = st.columns(6)
    k[0].metric("Records", f"{len(df):,}")
    k[1].metric("Countries", df["Area"].nunique())
    k[2].metric("Crops", df["Item"].nunique())
    k[3].metric("Years", f"{df['Year'].min()}–{df['Year'].max()}")
    k[4].metric("Mean yield", f"{df[TARGET].mean():,.0f}")
    k[5].metric("Median yield", f"{df[TARGET].median():,.0f}")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Data sample")
        st.dataframe(df.head(200), **STRETCH, height=330)
    with c2:
        st.subheader("Summary statistics")
        st.dataframe(df[NUM_COLS + [TARGET]].describe().T.round(2), **STRETCH, height=330)

    st.download_button("⬇️ Download filtered data (CSV)", df.to_csv(index=False).encode(),
                       "filtered_yield.csv", "text/csv")


# EDA

with tab_eda:
    st.subheader("Target distribution")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(df, x=TARGET, nbins=50, marginal="box", title="Yield distribution",
                           color_discrete_sequence=["#2e7d32"], labels=LABELS)
        st.plotly_chart(style(fig), **STRETCH)
        st.caption(f"Skewness: {df[TARGET].skew():.2f} (right-skewed)")
    with c2:
        order = df.groupby("Item")[TARGET].mean().sort_values(ascending=False).index
        fig = px.box(df, x="Item", y=TARGET, category_orders={"Item": list(order)},
                     color="Item", title="Yield by crop", color_discrete_sequence=COLOR_SEQ, labels=LABELS)
        fig.update_layout(showlegend=False, xaxis_tickangle=-40)
        st.plotly_chart(style(fig), **STRETCH)
    with c3:
        cm = df.groupby("Item")[TARGET].mean().sort_values().reset_index()
        fig = px.bar(cm, x=TARGET, y="Item", orientation="h", title="Mean yield by crop",
                     color=TARGET, color_continuous_scale="Greens", labels=LABELS)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig), **STRETCH)

    st.subheader("Numeric features")
    cols = st.columns(4)
    for col, name in zip(cols, NUM_COLS):
        fig = px.histogram(df, x=name, nbins=30, title=name, color_discrete_sequence=["#1976d2"], labels=LABELS)
        col.plotly_chart(style(fig, 300), **STRETCH)

    st.subheader("Relationship with yield")
    c1, c2, c3 = st.columns(3)
    sample = df.sample(min(len(df), 5000), random_state=1)
    for col, name in zip((c1, c2, c3), (RAIN, PEST, TEMP)):
        # CHANGED: Added labels, removed legend to declutter
        fig = px.scatter(sample, x=name, y=TARGET, color="Item", opacity=0.5,
                         title=f"{name} vs yield", color_discrete_sequence=COLOR_SEQ, labels=LABELS)
        fig.update_layout(showlegend=False) 
        col.plotly_chart(style(fig, 400), **STRETCH)
    st.caption("Scatter plots use a random sample of up to 5,000 rows for speed.")

    # CHANGED: Balanced the columns for heatmap and outliers
    c1, c2 = st.columns(2)
    with c1:
        corr = df[NUM_COLS + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Correlation heatmap")
        st.plotly_chart(style(fig, 420), **STRETCH)
    with c2:
        q1, q3 = df[TARGET].quantile([0.25, 0.75])
        iqr = q3 - q1
        out = df[(df[TARGET] < q1 - 1.5 * iqr) | (df[TARGET] > q3 + 1.5 * iqr)]
        st.markdown(f"**Outliers (IQR rule):** {len(out):,} rows ({len(out)/len(df)*100:.1f}%) - "
                    "mostly high-yield crops, so they're valid and kept.")
        oc = out["Item"].value_counts().reset_index()
        oc.columns = ["Item", "Outliers"]
        # CHANGED: Added explicit Y-axis title
        fig = px.bar(oc, x="Item", y="Outliers", title="Outliers by crop",
                     color_discrete_sequence=["#e65100"], labels={"Outliers": "Number of Outliers"})
        st.plotly_chart(style(fig, 340), **STRETCH)


# Countries & trends

with tab_geo:
    cmean = df.groupby("Area")[TARGET].mean().sort_values(ascending=False)
    n = st.slider("Countries to show in top/bottom charts", 5, 25, 10)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cmean.head(n).reset_index(), x="Area", y=TARGET,
                     title=f"Top {n} countries by mean yield", color_discrete_sequence=["#ef6c00"], labels=LABELS)
        st.plotly_chart(style(fig), **STRETCH)
    with c2:
        fig = px.bar(cmean.tail(n).reset_index(), x="Area", y=TARGET,
                     title=f"Bottom {n} countries by mean yield", color_discrete_sequence=["#c62828"], labels=LABELS)
        st.plotly_chart(style(fig), **STRETCH)

    c1, c2 = st.columns(2)
    with c1:
        ym = df.groupby("Year")[TARGET].mean().reset_index()
        fig = px.line(ym, x="Year", y=TARGET, markers=True, title="Average yield per year",
                      color_discrete_sequence=["#2e7d32"], labels=LABELS)
        st.plotly_chart(style(fig, 400), **STRETCH)
    with c2:
        yi = df.groupby(["Year", "Item"])[TARGET].mean().reset_index()
        fig = px.line(yi, x="Year", y=TARGET, color="Item", title="Average yield per year by crop",
                      color_discrete_sequence=COLOR_SEQ, labels=LABELS)
        st.plotly_chart(style(fig, 400), **STRETCH)

    st.subheader("Choropleth: mean yield by country")
    try:
        geo = df.groupby("Area")[TARGET].mean().reset_index()
        fig = px.choropleth(geo, locations="Area", locationmode="country names", color=TARGET,
                            color_continuous_scale="YlGn", title="Mean yield (hg/ha)", labels=LABELS)
        fig.update_geos(showframe=False, projection_type="natural earth")
        st.plotly_chart(style(fig, 480), **STRETCH)
        st.caption("Countries whose names don't match Plotly's country list are left blank.")
    except Exception as e:
        st.info(f"Map unavailable: {e}")

    c1, c2 = st.columns(2)
    with c1:
        rec = df["Area"].value_counts().head(15).reset_index()
        rec.columns = ["Area", "Records"]
        fig = px.bar(rec, x="Area", y="Records", title="Top 15 countries by number of records",
                     color_discrete_sequence=["#1976d2"], labels=LABELS)
        st.plotly_chart(style(fig), **STRETCH)
    with c2:
        heat = df.pivot_table(values=TARGET, index="Item", columns="Year", aggfunc="mean")
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlGn",
                        title="Mean yield: crop x year")
        st.plotly_chart(style(fig), **STRETCH)


# Model performance (uses the FULL dataset, not the filtered view)

with tab_model:
    pipe, metrics, y_te, pred, imp, combined = train_final(df_all)
    st.caption("Final model: Random Forest (200 trees), 80/20 split, trained on the full dataset "
               "(filters above don't affect the model).")
    m = st.columns(3)
    m[0].metric("Test R²", f"{metrics['R2']:.4f}")
    m[1].metric("MAE (hg/ha)", f"{metrics['MAE']:,.0f}")
    m[2].metric("RMSE (hg/ha)", f"{metrics['RMSE']:,.0f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        idx = np.random.RandomState(0).choice(len(y_te), min(4000, len(y_te)), replace=False)
        fig = px.scatter(x=y_te.iloc[idx], y=pred.iloc[idx], opacity=0.35,
                         labels={"x": "Actual", "y": "Predicted"}, title="Actual vs predicted",
                         color_discrete_sequence=["#1976d2"])
        lim = [float(min(y_te.min(), pred.min())), float(max(y_te.max(), pred.max()))]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect",
                                 line=dict(color="red", dash="dash")))
        st.plotly_chart(style(fig, 400), **STRETCH)
    with c2:
        fig = px.histogram(y_te - pred, nbins=60, title="Residuals (actual - predicted)",
                           color_discrete_sequence=["#6a1b9a"])
        fig.update_layout(showlegend=False, xaxis_title="Residual")
        st.plotly_chart(style(fig, 400), **STRETCH)
    with c3:
        comb = combined.sort_values().reset_index()
        comb.columns = ["Feature", "Importance"]
        fig = px.bar(comb, x="Importance", y="Feature", orientation="h",
                     title="Importance by original feature", color_discrete_sequence=["#00796b"])
        st.plotly_chart(style(fig, 400), **STRETCH)

    top = imp.sort_values(ascending=False).head(15).iloc[::-1].reset_index()
    top.columns = ["Feature", "Importance"]
    top["Feature"] = top["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
    fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                 title="Top 15 encoded features", color_discrete_sequence=["#00695c"])
    st.plotly_chart(style(fig, 420), **STRETCH)

    st.subheader("Model comparison")
    if st.button("Run comparison of all models"):
        st.session_state["cmp"] = compare_models(df_all)
    if "cmp" in st.session_state:
        cmp_df = st.session_state["cmp"]
        c1, c2 = st.columns([2, 3])
        c1.dataframe(cmp_df.round(3), **STRETCH, hide_index=True)
        melt = cmp_df.melt(id_vars="Model", value_vars=["Train R2", "Test R2"], var_name="Split",
                           value_name="R2")
        fig = px.bar(melt, x="Model", y="R2", color="Split", barmode="group",
                     title="Train vs test R² (gap = overfitting)",
                     color_discrete_sequence=["#90a4ae", "#2e7d32"])
        c2.plotly_chart(style(fig, 380), **STRETCH)


# Predict

with tab_pred:
    pipe, metrics, *_ = train_final(df_all)
    st.subheader("Enter conditions")

    with st.container(border=True):
        r1 = st.columns(3)
        area = r1[0].selectbox("Country", areas_all, index=0)
        item = r1[1].selectbox("Crop", crops_all, index=0)
        year = r1[2].number_input("Year", 1960, 2100, int(df_all["Year"].max()), step=1)
        r2 = st.columns(3)
        rain = r2[0].number_input("Average rainfall (mm/year)", 0.0, 10000.0,
                                  float(round(df_all[RAIN].median(), 0)), step=50.0)
        pest = r2[1].number_input("Pesticides (tonnes)", 0.0, 500000.0,
                                  float(round(df_all[PEST].median(), 0)), step=100.0)
        temp = r2[2].number_input("Average temperature (°C)", -10.0, 45.0,
                                  float(round(df_all[TEMP].median(), 1)), step=0.5)
        go_btn = st.button("🎯 Predict yield", type="primary", **STRETCH)

    def predict(**over):
        base = {"Year": year, RAIN: rain, PEST: pest, TEMP: temp, "Area": area, "Item": item}
        base.update(over)
        return float(pipe.predict(pd.DataFrame([base]))[0])

    if go_btn or "predicted" in st.session_state:
        if go_btn:
            st.session_state["predicted"] = True
        yhat = predict()
        crop_avg = df_all.loc[df_all["Item"] == item, TARGET].mean()
        ctry_avg = df_all.loc[(df_all["Item"] == item) & (df_all["Area"] == area), TARGET].mean()

        a, b, c, d = st.columns(4)
        a.metric("Predicted yield (hg/ha)", f"{yhat:,.0f}")
        b.metric("≈ tonnes / ha", f"{yhat/10000:,.2f}")
        c.metric(f"vs {item} global avg", f"{(yhat/crop_avg-1)*100:+.1f}%")
        d.metric(f"vs {area} avg" if not np.isnan(ctry_avg) else "vs country avg",
                 f"{(yhat/ctry_avg-1)*100:+.1f}%" if not np.isnan(ctry_avg) else "no history")

        if np.isnan(ctry_avg):
            st.warning(f"No '{item}' records for {area} in the training data - treat this as a rough estimate.")
        st.caption(f"Typical error of the model: ±{metrics['MAE']:,.0f} hg/ha (MAE on held-out data).")

        c1, c2, c3 = st.columns(3)
        with c1:
            vals = {"Prediction": yhat, f"{area} avg": ctry_avg, f"{item} global avg": crop_avg}
            vals = {k: v for k, v in vals.items() if not np.isnan(v)}
            fig = px.bar(x=list(vals.keys()), y=list(vals.values()), title="Prediction vs benchmarks",
                         color=list(vals.keys()), color_discrete_sequence=["#2e7d32", "#1976d2", "#90a4ae"])
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="hg/ha")
            st.plotly_chart(style(fig, 360), **STRETCH)
        with c2:
            grid = np.linspace(0, max(3000, rain * 2), 25)
            fig = px.line(x=grid, y=[predict(**{RAIN: g}) for g in grid], markers=True,
                          title="Sensitivity: rainfall", labels={"x": "Rainfall (mm)", "y": "Yield"},
                          color_discrete_sequence=["#1976d2"])
            fig.add_vline(x=rain, line_dash="dash", line_color="red")
            st.plotly_chart(style(fig, 360), **STRETCH)
        with c3:
            grid = np.linspace(float(df_all[TEMP].min()), float(df_all[TEMP].max()), 25)
            fig = px.line(x=grid, y=[predict(**{TEMP: g}) for g in grid], markers=True,
                          title="Sensitivity: temperature", labels={"x": "Temperature (°C)", "y": "Yield"},
                          color_discrete_sequence=["#ef6c00"])
            fig.add_vline(x=temp, line_dash="dash", line_color="red")
            st.plotly_chart(style(fig, 360), **STRETCH)

        st.subheader("Same conditions, all crops")
        allc = pd.DataFrame({"Item": crops_all,
                             "Predicted yield": [predict(Item=c) for c in crops_all]}) \
            .sort_values("Predicted yield", ascending=False)
        fig = px.bar(allc, x="Item", y="Predicted yield", title=f"Predicted yield by crop in {area}",
                     color="Predicted yield", color_continuous_scale="Greens")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 360), **STRETCH)