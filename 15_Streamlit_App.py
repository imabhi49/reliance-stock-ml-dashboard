import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go

from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Reliance ML Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM UI STYLE
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
    }

    .trade-card {
        border: 2px solid #2ecc71;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .no-trade-card {
        border: 2px solid #f39c12;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .small-note {
        font-size: 13px;
        opacity: 0.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_market_random_forest.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "final_market_random_forest_threshold.pkl"
)

BACKTEST_PATH = (
    BASE_DIR
    / "results"
    / "final_backtest_daily.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR
    / "results"
    / "final_test_predictions.csv"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# LOAD THRESHOLD
# =========================================================

if THRESHOLD_PATH.exists():

    threshold = float(
        joblib.load(
            THRESHOLD_PATH
        )
    )

else:

    threshold = 0.55


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📈 Model Control")

st.sidebar.write(
    "**Stock:** RELIANCE.NS"
)

st.sidebar.write(
    "**Market Context:** NIFTY 50"
)

st.sidebar.write(
    "**Final Model:** Random Forest"
)

st.sidebar.write(
    f"**Frozen Threshold:** {threshold:.2f}"
)

st.sidebar.markdown("---")

refresh_button = st.sidebar.button(
    "🔄 Refresh Market Data",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Research / educational dashboard. "
    "Backtesting results do not guarantee future performance."
)


# =========================================================
# MODEL CHECK
# =========================================================

if model is None:

    st.error(
        "Final model file was not found.\n\n"
        "Expected:\n"
        "models/final_market_random_forest.pkl"
    )

    st.stop()


# =========================================================
# DOWNLOAD MARKET DATA
# =========================================================

@st.cache_data(ttl=3600)
def download_market_data():

    reliance = yf.download(
        "RELIANCE.NS",
        start="2015-01-01",
        auto_adjust=False,
        progress=False
    )

    nifty = yf.download(
        "^NSEI",
        start="2015-01-01",
        auto_adjust=False,
        progress=False
    )

    if isinstance(
        reliance.columns,
        pd.MultiIndex
    ):

        reliance.columns = (
            reliance.columns
            .get_level_values(0)
        )

    if isinstance(
        nifty.columns,
        pd.MultiIndex
    ):

        nifty.columns = (
            nifty.columns
            .get_level_values(0)
        )

    return reliance, nifty


if refresh_button:

    st.cache_data.clear()


try:

    with st.spinner(
        "Downloading latest Reliance and NIFTY data..."
    ):

        reliance, nifty = download_market_data()

except Exception as error:

    st.error(
        f"Could not download market data: {error}"
    )

    st.stop()


if reliance.empty or nifty.empty:

    st.error(
        "Market data download returned no rows."
    )

    st.stop()


# =========================================================
# RELIANCE FEATURE ENGINEERING
# =========================================================

data = reliance.copy()

data["Daily_Return"] = (
    data["Close"].pct_change()
)

data["Return_Lag_1"] = (
    data["Daily_Return"].shift(1)
)

data["Return_Lag_2"] = (
    data["Daily_Return"].shift(2)
)

data["Return_Lag_3"] = (
    data["Daily_Return"].shift(3)
)

data["Return_Lag_5"] = (
    data["Daily_Return"].shift(5)
)


data["Return_5D"] = (
    data["Close"].pct_change(5)
)

data["Return_10D"] = (
    data["Close"].pct_change(10)
)

data["Return_20D"] = (
    data["Close"].pct_change(20)
)


data["SMA_5"] = (
    data["Close"]
    .rolling(5)
    .mean()
)

data["SMA_10"] = (
    data["Close"]
    .rolling(10)
    .mean()
)

data["SMA_20"] = (
    data["Close"]
    .rolling(20)
    .mean()
)


data["Close_SMA5_Ratio"] = (
    data["Close"]
    / data["SMA_5"]
)

data["Close_SMA10_Ratio"] = (
    data["Close"]
    / data["SMA_10"]
)

data["Close_SMA20_Ratio"] = (
    data["Close"]
    / data["SMA_20"]
)


data["EMA_12"] = (
    data["Close"]
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)

data["EMA_26"] = (
    data["Close"]
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)

data["MACD"] = (
    data["EMA_12"]
    - data["EMA_26"]
)

data["MACD_Signal"] = (
    data["MACD"]
    .ewm(
        span=9,
        adjust=False
    )
    .mean()
)

data["MACD_Hist"] = (
    data["MACD"]
    - data["MACD_Signal"]
)


def calculate_rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(period)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(period)
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss
    )

    return (
        100
        - (100 / (1 + rs))
    )


data["RSI_14"] = (
    calculate_rsi(
        data["Close"]
    )
)


data["Volatility_5"] = (
    data["Daily_Return"]
    .rolling(5)
    .std()
)

data["Volatility_10"] = (
    data["Daily_Return"]
    .rolling(10)
    .std()
)

data["Volatility_20"] = (
    data["Daily_Return"]
    .rolling(20)
    .std()
)


data["High_Low_Range"] = (
    (
        data["High"]
        - data["Low"]
    )
    / data["Close"]
)

data["Open_Close_Change"] = (
    (
        data["Close"]
        - data["Open"]
    )
    / data["Open"]
)


data["Volume_Change"] = (
    data["Volume"]
    .pct_change()
)

data["Volume_MA_5"] = (
    data["Volume"]
    .rolling(5)
    .mean()
)

data["Volume_Ratio_5"] = (
    data["Volume"]
    / data["Volume_MA_5"]
)


# =========================================================
# NIFTY FEATURE ENGINEERING
# =========================================================

nifty_data = nifty[
    [
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]
].copy()

nifty_data = (
    nifty_data
    .add_prefix("NIFTY_")
)


nifty_data["NIFTY_Return_1D"] = (
    nifty_data["NIFTY_Close"]
    .pct_change()
)

nifty_data["NIFTY_Return_3D"] = (
    nifty_data["NIFTY_Close"]
    .pct_change(3)
)

nifty_data["NIFTY_Return_5D"] = (
    nifty_data["NIFTY_Close"]
    .pct_change(5)
)

nifty_data["NIFTY_Return_10D"] = (
    nifty_data["NIFTY_Close"]
    .pct_change(10)
)

nifty_data["NIFTY_Return_20D"] = (
    nifty_data["NIFTY_Close"]
    .pct_change(20)
)


nifty_data["NIFTY_SMA_5"] = (
    nifty_data["NIFTY_Close"]
    .rolling(5)
    .mean()
)

nifty_data["NIFTY_SMA_20"] = (
    nifty_data["NIFTY_Close"]
    .rolling(20)
    .mean()
)


nifty_data["NIFTY_Close_SMA5_Ratio"] = (
    nifty_data["NIFTY_Close"]
    / nifty_data["NIFTY_SMA_5"]
)

nifty_data["NIFTY_Close_SMA20_Ratio"] = (
    nifty_data["NIFTY_Close"]
    / nifty_data["NIFTY_SMA_20"]
)


nifty_data["NIFTY_Volatility_5"] = (
    nifty_data["NIFTY_Return_1D"]
    .rolling(5)
    .std()
)

nifty_data["NIFTY_Volatility_10"] = (
    nifty_data["NIFTY_Return_1D"]
    .rolling(10)
    .std()
)

nifty_data["NIFTY_Volatility_20"] = (
    nifty_data["NIFTY_Return_1D"]
    .rolling(20)
    .std()
)


nifty_data["NIFTY_High_Low_Range"] = (
    (
        nifty_data["NIFTY_High"]
        - nifty_data["NIFTY_Low"]
    )
    / nifty_data["NIFTY_Close"]
)

nifty_data["NIFTY_Open_Close_Change"] = (
    (
        nifty_data["NIFTY_Close"]
        - nifty_data["NIFTY_Open"]
    )
    / nifty_data["NIFTY_Open"]
)


# =========================================================
# MERGE RELIANCE + NIFTY
# =========================================================

data = data.join(
    nifty_data,
    how="inner"
)


# =========================================================
# RELATIVE MARKET FEATURES
# =========================================================

data["REL_vs_NIFTY_1D"] = (
    data["Daily_Return"]
    - data["NIFTY_Return_1D"]
)

data["REL_vs_NIFTY_5D"] = (
    data["Return_5D"]
    - data["NIFTY_Return_5D"]
)

data["REL_vs_NIFTY_10D"] = (
    data["Return_10D"]
    - data["NIFTY_Return_10D"]
)

data["REL_vs_NIFTY_20D"] = (
    data["Return_20D"]
    - data["NIFTY_Return_20D"]
)


data["REL_NIFTY_Ratio"] = (
    data["Close"]
    / data["NIFTY_Close"]
)

data["REL_NIFTY_Ratio_Change_5D"] = (
    data["REL_NIFTY_Ratio"]
    .pct_change(5)
)

data["REL_NIFTY_Ratio_Change_20D"] = (
    data["REL_NIFTY_Ratio"]
    .pct_change(20)
)


data["REL_NIFTY_Corr_20"] = (
    data["Daily_Return"]
    .rolling(20)
    .corr(
        data["NIFTY_Return_1D"]
    )
)

data["REL_NIFTY_Corr_60"] = (
    data["Daily_Return"]
    .rolling(60)
    .corr(
        data["NIFTY_Return_1D"]
    )
)


# =========================================================
# EXACT FINAL MODEL FEATURES
# =========================================================

market_features = [

    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",

    "Return_5D",
    "Return_10D",
    "Return_20D",

    "Close_SMA5_Ratio",
    "Close_SMA10_Ratio",
    "Close_SMA20_Ratio",

    "MACD",
    "MACD_Hist",
    "RSI_14",

    "Volatility_5",
    "Volatility_10",
    "Volatility_20",

    "High_Low_Range",
    "Open_Close_Change",

    "Volume_Change",
    "Volume_Ratio_5",

    "NIFTY_Return_1D",
    "NIFTY_Return_3D",
    "NIFTY_Return_5D",
    "NIFTY_Return_10D",
    "NIFTY_Return_20D",

    "NIFTY_Close_SMA5_Ratio",
    "NIFTY_Close_SMA20_Ratio",

    "NIFTY_Volatility_5",
    "NIFTY_Volatility_10",
    "NIFTY_Volatility_20",

    "NIFTY_High_Low_Range",
    "NIFTY_Open_Close_Change",

    "REL_vs_NIFTY_1D",
    "REL_vs_NIFTY_5D",
    "REL_vs_NIFTY_10D",
    "REL_vs_NIFTY_20D",

    "REL_NIFTY_Ratio_Change_5D",
    "REL_NIFTY_Ratio_Change_20D",

    "REL_NIFTY_Corr_20",
    "REL_NIFTY_Corr_60"
]


# =========================================================
# CLEAN FEATURES
# =========================================================

feature_data = (
    data[market_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna()
)


if feature_data.empty:

    st.error(
        "No valid feature row could be generated."
    )

    st.stop()


latest_features = (
    feature_data
    .iloc[[-1]]
)


if hasattr(
    model,
    "feature_names_in_"
):

    latest_features = latest_features[
        model.feature_names_in_
    ]


latest_date = (
    latest_features.index[0]
)


# =========================================================
# FINAL PREDICTION
# =========================================================

prob_up = float(
    model.predict_proba(
        latest_features
    )[0, 1]
)

prediction = int(
    prob_up >= threshold
)


latest_close = float(
    data.loc[
        latest_date,
        "Close"
    ]
)

latest_nifty = float(
    data.loc[
        latest_date,
        "NIFTY_Close"
    ]
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "📈 Reliance ML Trading Dashboard"
)

st.caption(
    "Selective machine-learning trading signals using "
    "Reliance Industries and NIFTY 50 market context."
)

st.markdown("---")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "🎯 Prediction",
        "💰 Backtest",
        "🧠 Model Performance",
        "📘 Methodology"
    ]
)


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================

with tab1:

    st.subheader(
        "Latest Market Status"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Reliance Close",
        f"₹{latest_close:,.2f}"
    )

    col2.metric(
        "NIFTY 50",
        f"{latest_nifty:,.2f}"
    )

    col3.metric(
        "Probability of UP",
        f"{prob_up * 100:.2f}%"
    )

    col4.metric(
        "Decision Threshold",
        f"{threshold * 100:.0f}%"
    )


    if prediction == 1:

        st.markdown(
            """
            <div class="trade-card">
                TRADE SIGNAL
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="no-trade-card">
                NO TRADE
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write(
        f"Latest usable market date: "
        f"**{latest_date.date()}**"
    )


    st.subheader(
        "Reliance Price History"
    )

    price_chart = (
        data[["Close"]]
        .tail(250)
    )

    fig_price = go.Figure()

    fig_price.add_trace(
        go.Scatter(
            x=price_chart.index,
            y=price_chart["Close"],
            mode="lines",
            name="Reliance Close"
        )
    )

    fig_price.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True
    )


# =========================================================
# TAB 2 - PREDICTION
# =========================================================

with tab2:

    st.subheader(
        "Current Prediction"
    )

    if prediction == 1:

        st.success(
            f"UP probability = "
            f"{prob_up * 100:.2f}% "
            f"is above the frozen threshold "
            f"of {threshold * 100:.0f}%."
        )

    else:

        st.info(
            f"UP probability = "
            f"{prob_up * 100:.2f}% "
            f"is below the frozen threshold "
            f"of {threshold * 100:.0f}%."
        )


    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_up * 100,

            number={
                "suffix": "%"
            },

            title={
                "text":
                "Probability of Next-Day UP"
            },

            gauge={
                "axis": {
                    "range": [0, 100]
                },

                "threshold": {
                    "line": {
                        "width": 5
                    },

                    "value":
                        threshold * 100
                }
            }
        )
    )

    fig_gauge.update_layout(
        height=350
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )


    if PREDICTIONS_PATH.exists():

        predictions = pd.read_csv(
            PREDICTIONS_PATH,
            index_col="Date",
            parse_dates=True
        )

        st.subheader(
            "Historical Test Probabilities"
        )

        fig_prob = go.Figure()

        fig_prob.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["Prob_UP"],
                mode="lines",
                name="UP Probability"
            )
        )


        signal_dates = predictions[
            predictions["Predicted"] == 1
        ]

        fig_prob.add_trace(
            go.Scatter(
                x=signal_dates.index,
                y=signal_dates["Prob_UP"],
                mode="markers",
                name="Trade Signal",
                marker={
                    "size": 8
                }
            )
        )


        fig_prob.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text=(
                f"Threshold = "
                f"{threshold:.2f}"
            )
        )

        fig_prob.update_layout(
            height=450,
            xaxis_title="Date",
            yaxis_title="UP Probability",
            yaxis_range=[0, 1],
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )


# =========================================================
# TAB 3 - BACKTEST
# =========================================================

with tab3:

    st.subheader(
        "Out-of-Sample Backtesting"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Strategy Return",
        "+19.04%"
    )

    c2.metric(
        "Buy & Hold",
        "-4.58%"
    )

    c3.metric(
        "Trade Win Rate",
        "68.63%"
    )

    c4.metric(
        "Sharpe Ratio",
        "1.498"
    )


    c5, c6, c7 = (
        st.columns(3)
    )

    c5.metric(
        "Model Trades",
        "51"
    )

    c6.metric(
        "Winning Trades",
        "35"
    )

    c7.metric(
        "Maximum Drawdown",
        "-2.82%"
    )


    if BACKTEST_PATH.exists():

        backtest = pd.read_csv(
            BACKTEST_PATH,
            index_col="Date",
            parse_dates=True
        )


        st.subheader(
            "Equity Curve"
        )

        fig_equity = go.Figure()

        fig_equity.add_trace(
            go.Scatter(
                x=backtest.index,
                y=backtest[
                    "Strategy_Equity"
                ],
                mode="lines",
                name="Model Strategy"
            )
        )


        if (
            "Always_Trade_Equity"
            in backtest.columns
        ):

            fig_equity.add_trace(
                go.Scatter(
                    x=backtest.index,
                    y=backtest[
                        "Always_Trade_Equity"
                    ],
                    mode="lines",
                    name="Always Intraday"
                )
            )


        fig_equity.update_layout(
            height=450,
            xaxis_title="Date",
            yaxis_title="Growth of ₹1",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_equity,
            use_container_width=True
        )


        if "Strategy_Return_Net" in backtest.columns:

            st.subheader(
                "Strategy Drawdown"
            )

            running_peak = (
                backtest[
                    "Strategy_Equity"
                ]
                .cummax()
            )

            backtest["Drawdown"] = (
                backtest[
                    "Strategy_Equity"
                ]
                / running_peak
            ) - 1


            fig_dd = go.Figure()

            fig_dd.add_trace(
                go.Scatter(
                    x=backtest.index,
                    y=backtest[
                        "Drawdown"
                    ] * 100,
                    fill="tozeroy",
                    mode="lines",
                    name="Drawdown"
                )
            )

            fig_dd.update_layout(
                height=350,
                xaxis_title="Date",
                yaxis_title="Drawdown (%)"
            )

            st.plotly_chart(
                fig_dd,
                use_container_width=True
            )


# =========================================================
# TAB 4 - MODEL PERFORMANCE
# =========================================================

with tab4:

    st.subheader(
        "Final Unseen-Test Performance"
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    m1.metric(
        "Accuracy",
        "53.48%"
    )

    m2.metric(
        "Balanced Accuracy",
        "52.84%"
    )

    m3.metric(
        "ROC-AUC",
        "52.67%"
    )

    m4.metric(
        "UP Precision",
        "60.78%"
    )


    m5, m6 = (
        st.columns(2)
    )

    m5.metric(
        "UP Recall",
        "15.12%"
    )

    m6.metric(
        "Frozen Threshold",
        "0.55"
    )


    st.warning(
        """
        The classifier has modest overall
        next-day direction discrimination.

        The stronger project result comes from
        using only selective high-confidence
        trade signals rather than trading every day.
        """
    )


    st.subheader(
        "Confusion Matrix"
    )

    confusion_df = pd.DataFrame(
        [
            [192, 20],
            [174, 31]
        ],
        index=[
            "Actual DOWN",
            "Actual UP"
        ],
        columns=[
            "Predicted DOWN",
            "Predicted UP"
        ]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )


# =========================================================
# TAB 5 - METHODOLOGY
# =========================================================

with tab5:

    st.subheader(
        "Project Methodology"
    )

    st.markdown(
        """
        ### Prediction task

        Predict whether Reliance Industries will move
        **UP or DOWN on the next trading day**.

        ### Models evaluated

        - Logistic Regression
        - Random Forest
        - XGBoost
        - LSTM

        ### Final selected model

        **Market-context Random Forest**

        ### Random Forest configuration

        - `n_estimators = 200`
        - `max_depth = 5`
        - `min_samples_split = 10`
        - `min_samples_leaf = 5`
        - `max_features = "sqrt"`

        ### Market-context information

        The final model uses:

        - Reliance lagged returns
        - multi-day returns
        - moving-average ratios
        - MACD
        - RSI
        - volatility
        - intraday range
        - volume features
        - NIFTY 50 returns
        - NIFTY volatility
        - NIFTY trend
        - Reliance vs NIFTY relative returns
        - relative strength
        - rolling correlations

        ### Validation design

        - Chronological train / validation / test split
        - Walk-forward evaluation
        - Final untouched test period
        - Threshold selected only from development data

        ### Frozen trading threshold

        **0.55**

        A trade signal is generated only when:

        `Probability of UP >= 0.55`
        """
    )


    st.info(
        """
        Final unseen-test ROC-AUC was 0.5267.

        The model should therefore not be described as
        a highly accurate daily market-direction predictor.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Reliance ML Trading Dashboard | "
    "Research and educational use only. "
    "Historical backtesting does not guarantee future results."
)