import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. 系統環境配置
# ==============================================================================

st.set_page_config(page_title="David 乖離率線性回歸", layout="wide")

if "is_dark" not in st.session_state:
    st.session_state.is_dark = False
if "ma_period" not in st.session_state:
    st.session_state.ma_period = 260

# ==============================================================================
# 2. 視覺設計 Tokens（依 Design Handoff「Classical」設計系統規格）
# ==============================================================================

LIGHT_TOKENS = {
    "bg": "#f3f2f2",
    "surface_alt": "#f8f7f6",
    "text": "#201f1d",
    "text_muted": "rgba(32,31,29,0.55)",
    "divider": "rgba(32,31,29,0.16)",
    "accent": "#b68235",
    "grid_line": "rgba(32,31,29,0.08)",
    "shadow": "0 3px 10px rgba(45,43,43,0.14)",
    "lines": {
        "close": "#1a3a6b", "extreme_bull": "#8b1e1e", "bull": "#e2726b",
        "trend": "#d4a017", "bear": "#7fb88f", "extreme_bear": "#1f5c3d",
    },
}
DARK_TOKENS = {
    "bg": "#17140f",
    "surface_alt": "#1c1912",
    "text": "#f3ede2",
    "text_muted": "rgba(243,237,226,0.6)",
    "divider": "rgba(243,237,226,0.16)",
    "accent": "#c99a4e",
    "grid_line": "rgba(243,237,226,0.12)",
    "shadow": "0 12px 32px rgba(0,0,0,0.5)",
    "lines": {
        "close": "#5b9bf0", "extreme_bull": "#d0342c", "bull": "#f0918a",
        "trend": "#e6b325", "bear": "#8fd6a3", "extreme_bear": "#2f9d68",
    },
}

is_dark = st.session_state.is_dark
tok = DARK_TOKENS if is_dark else LIGHT_TOKENS
chart_template = "plotly_dark" if is_dark else "plotly_white"

# ==============================================================================
# 3. 全域樣式（字型／配色／按鈕／輸入框／側邊欄版面）
# ==============================================================================

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Lora:wght@400;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Lora', serif; }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
    }}

    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {tok['bg']} !important;
        color: {tok['text']} !important;
    }}
    [data-testid="stHeader"] {{ background-color: transparent !important; }}

    /* 側邊欄版面 */
    [data-testid="stSidebar"] {{
        background-color: {tok['bg']} !important;
        border-right: 1px solid {tok['divider']};
        min-width: 300px !important;
        max-width: 300px !important;
    }}
    [data-testid="stSidebarUserContent"] {{ padding: 40px 28px !important; }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 18px; }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-size: 12px !important;
        color: {tok['text_muted']} !important;
        font-family: 'Lora', serif !important;
    }}

    /* 輸入框：邊框只畫在最外層容器，內層元素一律去邊框/去底色，避免雙層邊框與底色不一致 */
    [data-testid="stSidebar"] div[data-baseweb="input"] {{
        border: 1px solid {tok['divider']} !important;
        border-radius: 4px !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] div[data-baseweb="input"] * {{
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] div[data-baseweb="input"] input {{
        color: {tok['text']} !important;
    }}

    /* 瀏覽器自動填入(autofill)會強制套用自己的底色，一般CSS蓋不掉，需用此專門技巧解除 */
    [data-testid="stSidebar"] input:-webkit-autofill,
    [data-testid="stSidebar"] input:-webkit-autofill:hover,
    [data-testid="stSidebar"] input:-webkit-autofill:focus,
    [data-testid="stSidebar"] input:-webkit-autofill:active {{
        -webkit-box-shadow: 0 0 0 1000px transparent inset !important;
        -webkit-text-fill-color: {tok['text']} !important;
        caret-color: {tok['text']};
        transition: background-color 9999s ease-in-out 0s;
    }}

    /* 按鈕：設計系統一律採 outline 樣式，不使用實心填色 */
    div.stButton > button {{
        width: 100%;
        background-color: transparent !important;
        border-radius: 4px !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: none !important;
    }}
    div.stButton > button[kind="secondary"] {{
        color: {tok['text']} !important;
        border: 1px solid {tok['divider']} !important;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        border-color: {tok['accent']} !important;
        color: {tok['accent']} !important;
    }}
    div.stButton > button[kind="primary"] {{
        color: {tok['accent']} !important;
        border: 1px solid {tok['accent']} !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: color-mix(in srgb, {tok['accent']} 12%, transparent) !important;
    }}

    /* 圖表卡片容器（st.container(border=True)） */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid {tok['divider']} !important;
        border-radius: 7px !important;
        background-color: {tok['surface_alt']} !important;
        box-shadow: {tok['shadow']};
    }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. 側邊欄：使用者參數輸入區
# ==============================================================================

st.sidebar.markdown(
    f"<div style='font-size:12px;letter-spacing:.08em;text-transform:uppercase;"
    f"color:{tok['text_muted']};margin-bottom:-6px;'>查詢設定</div>",
    unsafe_allow_html=True,
)

# 股票代號輸入
stock_id = st.sidebar.text_input("股票代號(如2330或AAPL)", "2330")

# 日期選擇
start_date = st.sidebar.date_input("起始日期", datetime(2019, 1, 1))
end_date = st.sidebar.date_input("結束日期", datetime.now())

st.sidebar.markdown(
    f"<hr style='border:none;border-top:1px solid {tok['divider']};margin:4px 0;'>",
    unsafe_allow_html=True,
)

# 移動平均線週期選擇：以按鈕切換 100 / 260 日
st.sidebar.markdown(
    f"<div style='font-size:12px;color:{tok['text_muted']};margin-bottom:-6px;'>移動平均線週期</div>",
    unsafe_allow_html=True,
)
ma_col1, ma_col2 = st.sidebar.columns(2)
ma100_clicked = ma_col1.button(
    "100 日", type=("primary" if st.session_state.ma_period == 100 else "secondary"),
    use_container_width=True,
)
ma260_clicked = ma_col2.button(
    "260 日", type=("primary" if st.session_state.ma_period == 260 else "secondary"),
    use_container_width=True,
)
if ma100_clicked:
    st.session_state.ma_period = 100
    st.rerun()
if ma260_clicked:
    st.session_state.ma_period = 260
    st.rerun()
ma_period = st.session_state.ma_period

st.sidebar.markdown(
    f"<hr style='border:none;border-top:1px solid {tok['divider']};margin:4px 0;'>",
    unsafe_allow_html=True,
)

# 圖表主題切換：以按鈕切換整頁亮/深色（對應網頁背景）
theme_label = "切換為深色" if not is_dark else "切換為亮色"
theme_icon = ":material/dark_mode:" if not is_dark else ":material/light_mode:"
theme_clicked = st.sidebar.button(theme_label, icon=theme_icon, type="secondary", use_container_width=True)

# 定義開始計算按鈕
calculate_btn = st.sidebar.button("開始計算", type="primary", use_container_width=True)

if theme_clicked:
    st.session_state.is_dark = not st.session_state.is_dark
    st.rerun()

# ==============================================================================
# 5. 主程式執行邏輯
# ==============================================================================

st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;">
      <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="{tok['accent']}" stroke-width="1.6">
        <path d="M3 17l5-6 4 3 6-9"/><path d="M14 5h5v5"/>
      </svg>
      <h1 style="margin:0;font-size:36px;">David 乖離率線性回歸</h1>
    </div>
    """, unsafe_allow_html=True)

if not calculate_btn:
    # 初始提示訊息
    st.markdown(
        f"<div style='color:{tok['text_muted']};font-size:15px;margin-top:12px;'>"
        f"請點開左上角選單 [ >> ] 在左側面板設定參數後，按「開始計算」即可產出圖表</div>",
        unsafe_allow_html=True,
    )
else:
    # 抓取資料：依序嘗試原始代號 → 加 .TW → 加 .TWO
    raw_id = stock_id.strip()
    candidates = [raw_id, f"{raw_id}.TW", f"{raw_id}.TWO"]
    for candidate in candidates:
        search_id = candidate
        data = yf.download(search_id, start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            break

    if not data.empty:
        # 顯示最終使用的股票代碼
        st.markdown(
            f"<div style='margin-top:4px;font-family:\"Cormorant Garamond\",serif;"
            f"font-size:19px;color:{tok['text_muted']};'>{search_id}</div>",
            unsafe_allow_html=True,
        )

        # --- 1. 處理 yfinance 可能產生的多層索引 (MultiIndex) ---
        # 如果欄位是多層的（例如包含 Ticker 名稱），則只取最內層的 Open, High, Low, Close
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 重設索引，將 Date 變成一個普通的欄位
        df = data.reset_index()

        # --- 相容不同版本 yfinance：reset_index() 後日期欄位可能為 'Date' 或 'Datetime' ---
        if 'Date' not in df.columns:
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'Date'})
            else:
                # 自動偵測 datetime 類型欄位並命名為 'Date'
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df = df.rename(columns={col: 'Date'})
                        break

        # --- 2. 核心修正：安全地建立運算用的欄位 ---
        # 直接從 df 中抓取欄位，避免使用 values.flatten() 導致的維度不符
        try:
            # 優先嘗試標準名稱
            df['Close_1D'] = df['Close']
            df['High_1D'] = df['High']
            df['Low_1D'] = df['Low']
            df['Open_1D'] = df['Open']
        except KeyError:
            # 如果抓不到 Close 欄位則停止執行並報錯
            st.error("找不到 'Close' 欄位，可能是資料下載格式不符，請重新嘗試。")
            st.stop()

        # [新增] 格式化日期字串，用於 X 軸顯示
        # --- 格式改為 YYYY-MM-DD ---
        # %Y (四位數年份), %m (兩位數月份), %d (兩位數日期)
        df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')

        # --- 3. 開始計算移動平均與乖離率 ---
        # A. 計算移動平均線 (MA)
        df['MA'] = df['Close_1D'].rolling(window=ma_period).mean()

        # B. 定義乖離率 (Bias Ratio)
        # 公式：(收盤價 / MA - 1) * 100
        df['Bias'] = ((df['Close_1D'] / df['MA']) - 1) * 100

        # --- 關鍵修正：同時處理 NaN 與 Inf (無限大) ---
        # 1. 將無限大替換為 NaN 2. 刪除所有 NaN
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Bias']).reset_index(drop=True)

        # 增加防錯機制：如果過濾後資料太少，則不進行回歸
        if len(df) < 10:
            st.error(f"❌ 目前日期範圍內的有效資料太少（少於 10 筆），無法進行 {ma_period} 天回歸分析。請加長起始日期。")
            st.stop()

        # C. 線性回歸計算 (針對乖離率)
        # X 為時間索引，Y 為乖離率
        X = np.array(df.index).reshape(-1, 1)
        Y = df['Bias'].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, Y)

        # 乖離率回歸值 (Middle Line)
        df['Bias_Reg'] = model.predict(X)

        # D. 計算離差與標準差 (SD)
        # 離差 = 實際乖離率 - 回歸值
        df['Deviation'] = df['Bias'] - df['Bias_Reg']
        sd_val = df['Deviation'].std()

        # E. 計算五線譜軌道 (基於乖離率回歸)
        df['Bias_P2SD'] = df['Bias_Reg'] + (2 * sd_val)  # 極端樂觀 (+2SD)
        df['Bias_P1SD'] = df['Bias_Reg'] + sd_val        # 樂觀 (+1SD)
        df['Bias_M1SD'] = df['Bias_Reg'] - sd_val        # 悲觀 (-1SD)
        df['Bias_M2SD'] = df['Bias_Reg'] - (2 * sd_val)  # 極端悲觀 (-2SD)

        # F. 繪圖：使用 Plotly（配色依 Design Handoff tokens）
        fig = go.Figure()
        lines = tok['lines']

        # 1. 實際乖離率曲線 (主線)
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias'], name='實際乖離率',
                                  line=dict(color=lines['close'], width=2)))

        # 2. 線性回歸線 (中心線)
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_Reg'], name='回歸中線',
                                  line=dict(color=lines['trend'], width=2)))

        # 3. 標準差軌道線
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_P2SD'], name='+2SD 極端樂觀',
                                  line=dict(color=lines['extreme_bull'], width=1.6, dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_P1SD'], name='+1SD 樂觀',
                                  line=dict(color=lines['bull'], width=1.6, dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_M1SD'], name='-1SD 悲觀',
                                  line=dict(color=lines['bear'], width=1.6, dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_M2SD'], name='-2SD 極端悲觀',
                                  line=dict(color=lines['extreme_bear'], width=1.6, dash='dash')))

        # 圖表佈局設定
        fig.update_layout(
            height=600,
            template=chart_template,
            hovermode="x unified",
            paper_bgcolor=tok['surface_alt'],
            plot_bgcolor=tok['surface_alt'],
            font=dict(color=tok['text'], family='Lora, serif', size=14),

            xaxis=dict(
                type='category',
                color=tok['text'],
                tickfont=dict(color=tok['text'], size=12),
                title=dict(text="日期", font=dict(color=tok['text'], size=14)),
                nticks=8,
                gridcolor=tok['grid_line'],
                zeroline=True,
                zerolinecolor=tok['grid_line'],
                zerolinewidth=1,
            ),

            yaxis=dict(
                color=tok['text'],
                tickfont=dict(color=tok['text'], size=12),
                title=dict(text="乖離率 (%)", font=dict(color=tok['text'], size=14)),
                gridcolor=tok['grid_line'],
                zeroline=True,
                zerolinecolor=tok['grid_line'],
                zerolinewidth=1,
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color=tok['text_muted'], family='Lora, serif', size=12),
            )
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

        # G. 顯示最後數據數據摘要
        st.markdown("<h4 style='margin:28px 0 16px;'>最後交易日數據摘要</h4>", unsafe_allow_html=True)
        last_row = df.iloc[-1]

        def _stat_card(label, value):
            return f"""
            <div style="border:1px solid {tok['divider']};border-radius:4px;padding:20px 22px;">
              <div style="font-size:12px;color:{tok['text_muted']};margin-bottom:8px;">{label}</div>
              <div style="font-family:'Cormorant Garamond',serif;font-size:28px;font-variant-numeric:tabular-nums;">{value}</div>
            </div>
            """

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(_stat_card("最後收盤價", f"{last_row['Close_1D']:.2f}"), unsafe_allow_html=True)
        col2.markdown(_stat_card("目前乖離率", f"{last_row['Bias']:.2f}%"), unsafe_allow_html=True)
        col3.markdown(_stat_card("回歸中線值", f"{last_row['Bias_Reg']:.2f}%"), unsafe_allow_html=True)
        col4.markdown(_stat_card("標準差 (SD)", f"{sd_val:.2f}%"), unsafe_allow_html=True)

    else:
        st.error(f"找不到股票資料（已嘗試：{', '.join(candidates)}），請檢查代號或日期。")
