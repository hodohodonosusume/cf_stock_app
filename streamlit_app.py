import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ===========================
# ページ設定
# ===========================
st.set_page_config(
    page_title="📊 株式AI分析ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# ユーティリティ
# ===========================
def safe_code_str(x) -> str:
    """コード列を安全に4桁文字列にする（数値/文字列/NaN対応）"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # 末尾に余計な文字が付いている場合でも、先頭の数字部分を優先
    # 例: "130A" → "130"
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            break
    if num == "":
        num = s  # それでもダメなら元文字列
    return num.zfill(4)

# ===========================
# キャッシング用の関数
# ===========================
@st.cache_resource
def load_stock_master():
    """株式マスタデータを読み込み（.xlsx優先）"""
    try:
        if os.path.exists("stock_all.xlsx"):
            df = pd.read_excel("stock_all.xlsx")
        elif os.path.exists("stock_all.xls"):
            df = pd.read_excel("stock_all.xls")
        else:
            return None

        # コード列を安全な文字列4桁に正規化
        if "コード" in df.columns:
            df["コード"] = df["コード"].apply(safe_code_str)
        return df
    except Exception as e:
        st.error(f"株式マスタデータの読み込みエラー: {e}")
        return None


@st.cache_resource
def load_predictor():
    """AI予測モデルを読み込み（今はダミー）"""
    try:
        if os.path.exists("selected_advanced_vwap_indicators_model.pkl"):
            with open("selected_advanced_vwap_indicators_model.pkl", "rb") as f:
                return pickle.load(f)
        elif os.path.exists("selected_advanced_vwap_indicators_model.txt"):
            st.info("モデルをテキストから読み込み中...")
            # ここではプレースホルダとして文字列を返す
            return "model_loaded"
    except Exception as e:
        st.warning(f"AI予測モデルの読み込みに失敗: {e}")
    return None


@st.cache_data(ttl=3600)
def get_chart_data(
    code: str, interval: str = "1d", period: str = "1y", max_bars: int = 90
):
    """銘柄チャートデータを取得"""
    try:
        interval_map = {
            "1d": ("1y", "1d"),
            "1w": ("2y", "1wk"),
            "1mo": ("5y", "1mo"),
        }
        if interval not in interval_map:
            interval = "1d"

        period_days = {
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }
        days = period_days.get(period, 365)
        yf_period, yf_interval = interval_map[interval]

        symbol = f"{code}.T"
        stock = yf.Ticker(symbol)
        df = stock.history(period=yf_period, interval=yf_interval)

        if not df.empty:
            # ローソク足に必要な列だけ残す
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            # 期間に合わせて末尾N本だけにトリミング
            if interval == "1d":
                bars = days
            elif interval == "1w":
                bars = days // 5
            else:
                bars = days // 20
            bars = min(bars, max_bars)
            return df.tail(bars)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"チャートデータ取得エラー ({code}): {e}")
        return pd.DataFrame()


@st.cache_data
def get_industries(df: pd.DataFrame):
    if df is not None and "33業種区分" in df.columns:
        return sorted(df["33業種区分"].dropna().unique().tolist())
    return []


@st.cache_data
def get_sizes(df: pd.DataFrame):
    if df is not None and "規模区分" in df.columns:
        return sorted(df["規模区分"].dropna().unique().tolist())
    return []


def get_stocks_by_industry(df: pd.DataFrame, industry: str):
    if df is None or "33業種区分" not in df.columns:
        return []
    sub = df[df["33業種区分"] == industry]
    return [
        {"code": safe_code_str(row["コード"]), "name": row["銘柄名"]}
        for _, row in sub.iterrows()
    ]


def get_stocks_by_size(df: pd.DataFrame, size: str):
    if df is None or "規模区分" not in df.columns:
        return []
    sub = df[df["規模区分"] == size]
    return [
        {"code": safe_code_str(row["コード"]), "name": row["銘柄名"]}
        for _, row in sub.iterrows()
    ]


@st.cache_data(ttl=3600)
def get_index_data(code: str):
    """インデックス・為替データを取得"""
    mapping = {
        "nikkei": "^N225",
        "topix": "^TOPX",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "vix": "^VIX",
        "jpy_usd": "JPY=X",
        "eur_jpy": "EURJPY=X",
    }
    symbol = mapping.get(code)
    if symbol is None:
        return pd.DataFrame()
    try:
        df = yf.Ticker(symbol).history(period="1y", interval="1d")
        if df.empty:
            return pd.DataFrame()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.warning(f"インデックスデータ取得エラー ({code}): {e}")
        return pd.DataFrame()


def get_aggregate_data(codes, period: str = "1y"):
    """複数銘柄の平均終値を算出"""
    codes = [c for c in codes if c]
    if len(codes) < 2:
        return pd.DataFrame()

    end = datetime.now()
    start = end - timedelta(days=365)

    common_index = None
    price_matrix = []

    for code in codes:
        try:
            symbol = f"{code}.T"
            df = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
            if df.empty:
                continue
            df = df[["Close"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
            price_matrix.append(df)
        except Exception:
            continue

    if not price_matrix or common_index is None:
        return pd.DataFrame()

    closes = [df.reindex(common_index)["Close"].values for df in price_matrix]
    closes = np.vstack(closes)
    mean_price = closes.mean(axis=0)

    result = pd.DataFrame({"Close": mean_price}, index=common_index)
    return result


def get_ai_prediction(code: str):
    """簡易AI予測（過去30日の平均変化率から5営業日後を推定）"""
    try:
        symbol = f"{code}.T"
        df = yf.Ticker(symbol).history(period="1y", interval="1d")
        if df.empty:
            return None
        recent = df["Close"].tail(30)
        if len(recent) < 2:
            return None
        pct_change = recent.pct_change().dropna().mean()
        current_price = df["Close"].iloc[-1]
        predicted_price = current_price * (1 + pct_change * 5)
        confidence = float(min(abs(pct_change) * 100, 80))
        return {
            "code": code,
            "current": float(current_price),
            "predicted": float(predicted_price),
            "change_pct": float((predicted_price / current_price - 1) * 100),
            "confidence": confidence,
        }
    except Exception:
        return None


def plot_candlestick(df: pd.DataFrame, title: str = ""):
    return build_candlestick_figure(df, title, [])


def add_moving_average(df: pd.DataFrame, window: int, kind: str = "sma"):
    if kind == "ema":
        return df["Close"].ewm(span=window, adjust=False).mean()
    if kind == "wma":
        weights = np.arange(1, window + 1)
        return (
            df["Close"]
            .rolling(window)
            .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        )
    return df["Close"].rolling(window).mean()


def add_rsi(df: pd.DataFrame, window: int = 14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    percent_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(d).mean()
    return percent_k, percent_d


def add_atr(df: pd.DataFrame, window: int = 14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def add_adx(df: pd.DataFrame, window: int = 14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = add_atr(df, 1)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / tr.rolling(
        window
    ).sum()
    minus_di = (
        100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / tr.rolling(window).sum()
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx


def add_obv(df: pd.DataFrame):
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def add_vwap(df: pd.DataFrame):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()


def add_ichimoku(df: pd.DataFrame):
    conversion = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    base = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
    span_a = ((conversion + base) / 2).shift(26)
    span_b = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2).shift(26)
    lagging = df["Close"].shift(-26)
    return conversion, base, span_a, span_b, lagging


def build_candlestick_figure(df: pd.DataFrame, title: str, indicators: list[str]):
    if df is None or df.empty:
        return None

    has_oscillator = any(
        ind in indicators
        for ind in ["RSI(14)", "MACD", "Stochastic", "ATR(14)", "ADX(14)", "OBV"]
    )
    rows = 3 if has_oscillator else 2
    row_heights = [0.6, 0.2, 0.2] if rows == 3 else [0.7, 0.3]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="価格",
        ),
        row=1,
        col=1,
    )

    color_up = "#26a69a"
    color_down = "#ef5350"
    volume_colors = np.where(df["Close"] >= df["Open"], color_up, color_down)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=volume_colors,
            name="出来高",
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    def add_price_trace(series, name, line=None):
        fig.add_trace(
            go.Scatter(x=df.index, y=series, mode="lines", name=name, line=line),
            row=1,
            col=1,
        )

    if "SMA(20)" in indicators:
        add_price_trace(add_moving_average(df, 20, "sma"), "SMA20", {"width": 1.5})
    if "SMA(50)" in indicators:
        add_price_trace(add_moving_average(df, 50, "sma"), "SMA50", {"width": 1.5})
    if "EMA(20)" in indicators:
        add_price_trace(add_moving_average(df, 20, "ema"), "EMA20", {"width": 1.2})
    if "WMA(20)" in indicators:
        add_price_trace(add_moving_average(df, 20, "wma"), "WMA20", {"width": 1.2})
    if "VWAP" in indicators:
        add_price_trace(add_vwap(df), "VWAP", {"width": 1.2})
    if "Bollinger(20)" in indicators:
        sma = add_moving_average(df, 20, "sma")
        std = df["Close"].rolling(20).std()
        add_price_trace(sma + std * 2, "Bollinger Upper", {"dash": "dash"})
        add_price_trace(sma - std * 2, "Bollinger Lower", {"dash": "dash"})
    if "Ichimoku" in indicators:
        conv, base, span_a, span_b, lagging = add_ichimoku(df)
        add_price_trace(conv, "転換線", {"width": 1.2})
        add_price_trace(base, "基準線", {"width": 1.2})
        add_price_trace(span_a, "先行スパンA", {"dash": "dot"})
        add_price_trace(span_b, "先行スパンB", {"dash": "dot"})
        add_price_trace(lagging, "遅行スパン", {"dash": "dash"})

    if has_oscillator:
        osc_row = 3
        if "RSI(14)" in indicators:
            fig.add_trace(
                go.Scatter(x=df.index, y=add_rsi(df), mode="lines", name="RSI(14)"),
                row=osc_row,
                col=1,
            )
        if "MACD" in indicators:
            macd, signal, hist = add_macd(df)
            fig.add_trace(
                go.Scatter(x=df.index, y=macd, mode="lines", name="MACD"),
                row=osc_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=signal, mode="lines", name="Signal"),
                row=osc_row,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=df.index, y=hist, name="MACD Hist", opacity=0.4),
                row=osc_row,
                col=1,
            )
        if "Stochastic" in indicators:
            percent_k, percent_d = add_stochastic(df)
            fig.add_trace(
                go.Scatter(x=df.index, y=percent_k, mode="lines", name="%K"),
                row=osc_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=percent_d, mode="lines", name="%D"),
                row=osc_row,
                col=1,
            )
        if "ATR(14)" in indicators:
            fig.add_trace(
                go.Scatter(x=df.index, y=add_atr(df), mode="lines", name="ATR(14)"),
                row=osc_row,
                col=1,
            )
        if "ADX(14)" in indicators:
            fig.add_trace(
                go.Scatter(x=df.index, y=add_adx(df), mode="lines", name="ADX(14)"),
                row=osc_row,
                col=1,
            )
        if "OBV" in indicators:
            fig.add_trace(
                go.Scatter(x=df.index, y=add_obv(df), mode="lines", name="OBV"),
                row=osc_row,
                col=1,
            )

    fig.update_layout(
        title=title,
        yaxis_title="株価 (円)",
        xaxis_title="日付",
        template="plotly_dark",
        height=650 if rows == 3 else 560,
        hovermode="x unified",
        dragmode="pan",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
        uirevision="chart",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
    )
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig


def plot_line(df: pd.DataFrame, title: str = "", column: str = "Close"):
    if df is None or df.empty or column not in df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[column], mode="lines", name=column))
    fig.update_layout(
        title=title,
        yaxis_title="値",
        xaxis_title="日付",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    return fig


# ===========================
# セッションステート初期化
# ===========================
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []
if "run_ai_analysis" not in st.session_state:
    st.session_state.run_ai_analysis = False

# ===========================
# メインUI
# ===========================
st.title("📊 株式AI分析ツール")
st.markdown("高度なテクニカル分析とAI予測を組み合わせた株式分析プラットフォーム")

stock_master_df = load_stock_master()
predictor = load_predictor()

if stock_master_df is None:
    st.error("❌ 株式マスタデータが見つかりません。stock_all.xlsx または stock_all.xls を配置してください。")
    st.stop()

# ===========================
# サイドバー: 銘柄選択
# ===========================
with st.sidebar:
    st.header("銘柄選択")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("選択中", len(st.session_state.selected_stocks), "銘柄")
    with col2:
        if st.button("🗑️ 全クリア"):
            st.session_state.selected_stocks = []
            st.session_state.run_ai_analysis = False
            st.experimental_rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["全銘柄", "業種で選択", "規模で選択", "直接入力"])

    # 全銘柄タブ
    with tab1:
        st.subheader("全銘柄から検索")
        all_stocks = []
        if "コード" in stock_master_df.columns and "銘柄名" in stock_master_df.columns:
            for _, row in stock_master_df.iterrows():
                code = safe_code_str(row["コード"])
                name = row["銘柄名"]
                if code:
                    all_stocks.append({"code": code, "name": name})

        search_text = st.text_input("銘柄名またはコードで検索", key="search_all")
        filtered = [
            s
            for s in all_stocks
            if search_text.lower() in str(s["name"]).lower()
            or search_text in s["code"]
        ]

        for stock in filtered[:50]:
            label = f"{stock['code']} - {stock['name']}"
            key = f"all_{stock['code']}"
            checked = stock["code"] in st.session_state.selected_stocks
            new_val = st.checkbox(label, value=checked, key=key)
            if new_val and stock["code"] not in st.session_state.selected_stocks:
                st.session_state.selected_stocks.append(stock["code"])

    # 業種タブ
    with tab2:
        st.subheader("業種で選択")
        industries = get_industries(stock_master_df)
        selected_industry = st.selectbox(
            "33業種区分を選択", ["-- 業種を選択 --"] + industries, key="industry_select"
        )
        if selected_industry != "-- 業種を選択 --":
            stocks_by_industry = get_stocks_by_industry(stock_master_df, selected_industry)
            st.info(f"この業種には {len(stocks_by_industry)} 銘柄あります")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("この業種を全て追加"):
                    for s in stocks_by_industry:
                        if s["code"] not in st.session_state.selected_stocks:
                            st.session_state.selected_stocks.append(s["code"])
                    st.success(f"{len(stocks_by_industry)}銘柄を追加しました")

    # 規模タブ
    with tab3:
        st.subheader("規模で選択")
        sizes = get_sizes(stock_master_df)
        selected_size = st.selectbox(
            "規模区分を選択", ["-- 規模を選択 --"] + sizes, key="size_select"
        )
        if selected_size != "-- 規模を選択 --":
            stocks_by_size = get_stocks_by_size(stock_master_df, selected_size)
            st.info(f"この規模には {len(stocks_by_size)} 銘柄あります")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("この規模を全て追加"):
                    for s in stocks_by_size:
                        if s["code"] not in st.session_state.selected_stocks:
                            st.session_state.selected_stocks.append(s["code"])
                    st.success(f"{len(stocks_by_size)}銘柄を追加しました")

    # 直接入力タブ
    with tab4:
        st.subheader("直接入力")
        manual_codes = st.text_area(
            "銘柄コードをカンマ区切りで入力（例：1301,1305,7203）"
        )
        if st.button("追加"):
            codes = [
                safe_code_str(c)
                for c in manual_codes.split(",")
                if safe_code_str(c) != ""
            ]
            added = 0
            for code in codes:
                if code not in st.session_state.selected_stocks:
                    st.session_state.selected_stocks.append(code)
                    added += 1
            if added > 0:
                st.success(f"{added}銘柄を追加しました")

    st.divider()
    if st.button("🤖 AI分析実行", use_container_width=True):
        st.session_state.run_ai_analysis = True

# ===========================
# メインタブ
# ===========================
tab_chart, tab_index, tab_agg, tab_ai = st.tabs(
    ["📈 チャート", "📊 指数・為替", "📉 平均インデックス", "🤖 AI予測"]
)

# ---- チャートタブ ----
with tab_chart:
    if not st.session_state.selected_stocks:
        st.info("📌 左のサイドバーから銘柄を選択してください")
    else:
        st.subheader(f"チャート（選択中: {len(st.session_state.selected_stocks)} 銘柄）")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_cols = st.selectbox("表示列", [1, 2, 3, 4], index=1)
        with col2:
            interval = st.selectbox(
                "足",
                ["1d", "1w", "1mo"],
                index=0,
                format_func=lambda x: {"1d": "日足", "1w": "週足", "1mo": "月足"}[x],
            )
        with col3:
            period = st.selectbox(
                "期間",
                ["3mo", "6mo", "1y", "2y", "5y"],
                index=2,
                format_func=lambda x: {
                    "3mo": "3ヶ月",
                    "6mo": "6ヶ月",
                    "1y": "1年",
                    "2y": "2年",
                    "5y": "5年",
                }[x],
            )
        with col4:
            if st.button("🔄 キャッシュクリア", use_container_width=True):
                st.cache_data.clear()
                st.experimental_rerun()

        indicator_options = [
            "SMA(20)",
            "SMA(50)",
            "EMA(20)",
            "WMA(20)",
            "VWAP",
            "Bollinger(20)",
            "Ichimoku",
            "RSI(14)",
            "MACD",
            "Stochastic",
            "ATR(14)",
            "ADX(14)",
            "OBV",
        ]
        selected_indicators = st.multiselect(
            "テクニカル指標",
            indicator_options,
            default=["SMA(20)", "EMA(20)", "RSI(14)"],
        )
        st.caption("ローソク足は最大90本まで表示されます。")

        charts = []
        for code in st.session_state.selected_stocks:
            df_c = get_chart_data(code, interval, period, max_bars=90)
            if df_c is not None and not df_c.empty:
                row = stock_master_df[stock_master_df["コード"] == code]
                name = row["銘柄名"].iloc[0] if not row.empty else ""
                fig = build_candlestick_figure(
                    df_c, f"{code} {name}", selected_indicators
                )
                charts.append(fig)

        if not charts:
            st.warning("選択された銘柄のチャートデータが取得できませんでした。")
        else:
            for i in range(0, len(charts), display_cols):
                cols = st.columns(display_cols)
                for j, fig in enumerate(charts[i : i + display_cols]):
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)

# ---- 指数・為替タブ ----
with tab_index:
    st.subheader("指数・為替チャート")
    indices = {
        "nikkei": "📈 日経平均（Nikkei 225）",
        "topix": "📊 TOPIX",
        "sp500": "🇺🇸 S&P 500",
        "nasdaq": "🖥 NASDAQ",
        "vix": "😨 VIX指数",
        "jpy_usd": "💱 USD/JPY",
        "eur_jpy": "💶 EUR/JPY",
    }
    selected_indices = st.multiselect(
        "表示する指数を選択",
        list(indices.keys()),
        default=["nikkei", "topix"],
        format_func=lambda x: indices[x],
    )
    for key in selected_indices:
        df_i = get_index_data(key)
        if df_i is not None and not df_i.empty:
            fig = plot_line(df_i, indices[key], "Close")
            st.plotly_chart(fig, use_container_width=True)

# ---- 平均インデックスタブ ----
with tab_agg:
    st.subheader("📊 選択銘柄の平均インデックス")
    if len(st.session_state.selected_stocks) < 2:
        st.info("2つ以上の銘柄を選択してください。")
    else:
        df_agg = get_aggregate_data(st.session_state.selected_stocks)
        if df_agg is None or df_agg.empty:
            st.warning("平均データが計算できませんでした。")
        else:
            fig = plot_line(
                df_agg,
                f"{len(st.session_state.selected_stocks)}銘柄の平均株価",
                "Close",
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            now = df_agg["Close"].iloc[-1]
            first = df_agg["Close"].iloc[0]
            with col1:
                st.metric("現在値", f"¥{now:.2f}")
            with col2:
                st.metric("変化額", f"¥{now - first:.2f}")
            with col3:
                chg_pct = (now / first - 1) * 100
                st.metric("変化率", f"{chg_pct:.2f}%")
            with col4:
                st.metric("最高値", f"¥{df_agg['Close'].max():.2f}")

# ---- AI予測タブ ----
with tab_ai:
    st.subheader("🤖 AI予測分析")

    if st.button("🔍 予測を実行", use_container_width=True):
        st.session_state.run_ai_analysis = True

    if st.session_state.run_ai_analysis:
        if not st.session_state.selected_stocks:
            st.error("❌ 銘柄を選択してください。")
        else:
            st.info(f"🔄 {len(st.session_state.selected_stocks)} 銘柄を分析中...")
            predictions = []
            progress = st.progress(0.0)
            total = len(st.session_state.selected_stocks)

            for idx, code in enumerate(st.session_state.selected_stocks):
                pred = get_ai_prediction(code)
                if pred is not None:
                    predictions.append(pred)
                progress.progress((idx + 1) / total)

            if not predictions:
                st.error("❌ 予測データを取得できませんでした。")
            else:
                df_pred = pd.DataFrame(predictions)

                def get_stock_name(c):
                    row = stock_master_df[stock_master_df["コード"] == c]
                    return row["銘柄名"].iloc[0] if not row.empty else ""

                df_pred["銘柄名"] = df_pred["code"].apply(get_stock_name)
                df_display = df_pred[
                    ["code", "銘柄名", "current", "predicted", "change_pct", "confidence"]
                ].copy()
                df_display.columns = [
                    "コード",
                    "銘柄名",
                    "現在値",
                    "予想値",
                    "変化率(%)",
                    "信頼度(%)",
                ]
                df_display["現在値"] = df_display["現在値"].apply(
                    lambda x: f"¥{x:.0f}"
                )
                df_display["予想値"] = df_display["予想値"].apply(
                    lambda x: f"¥{x:.0f}"
                )
                df_display["変化率(%)"] = df_display["変化率(%)"].apply(
                    lambda x: f"{x:+.2f}%"
                )
                df_display["信頼度(%)"] = df_display["信頼度(%)"].apply(
                    lambda x: f"{x:.1f}%"
                )

                st.dataframe(df_display, use_container_width=True)

                col_u, col_d = st.columns(2)
                up_count = (df_pred["change_pct"] > 0).sum()
                down_count = (df_pred["change_pct"] < 0).sum()
                with col_u:
                    st.metric("📈 上昇予想", f"{up_count}銘柄")
                with col_d:
                    st.metric("📉 下降予想", f"{down_count}銘柄")

# ===========================
# フッター
# ===========================
st.divider()
st.markdown(
    """
### 📌 使用方法
1. 左のサイドバーから銘柄を選択
2. チャートタブで価格推移を確認
3. 指数・為替タブで市場全体を確認
4. 平均インデックスタブでポートフォリオ感覚の動きを確認
5. AI予測タブで今後の方向性を参考にする

### ⚠️ 免責事項
このツールはあくまで分析ツールです。投資判断はご自身の責任で行ってください。
"""
)
