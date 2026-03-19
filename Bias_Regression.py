import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --- 1. 網頁配置與表頭 ---
st.set_page_config(page_title="David 乖離率線性回歸", layout="wide")

# --- 2. 側邊欄：查詢設定 ---
st.sidebar.header("查詢設定")

# 股票代號輸入
stock_id = st.sidebar.text_input("股票代號(如2330.TW或AAPL)", "2330.TW")

# 日期選擇
start_date = st.sidebar.date_input("起始日期", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("結束日期", datetime.now())

# 移動平均線週期選擇
ma_period = st.sidebar.radio("移動平均線週期", [100, 265], index=0)

# 圖表主題選擇
theme_choice = st.sidebar.radio("圖表主題(對應網頁背景)", ["亮色(白色背景)", "深色(深色背景)"])

# --- 3. 強制背景色切換邏輯 (CSS) ---
# 此段確保 Streamlit 元件顏色與圖表主題同步
if theme_choice == "深色(深色背景)":
    chart_template = "plotly_dark"
    font_color = "white"
    bg_color = "#0E1117"
    st.markdown("""
        <style>
        /* 強制側邊欄、主背景、文字顏色為深色 */
        [data-testid="stSidebar"], .stApp, header { background-color: #0E1117 !important; color: white !important; }
        .stMarkdown, p, h1, h2, h3, span { color: white !important; }
        /* 調整輸入框文字顏色 */
        input { color: white !important; background-color: #262730 !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    chart_template = "plotly_white"
    font_color = "black"
    bg_color = "#FFFFFF"
    st.markdown("""
        <style>
        /* 1. 強制背景與文字顏色 */
        [data-testid="stSidebar"], .stApp, header { 
            background-color: #FFFFFF !important; 
            color: black !important; 
        }
        .stMarkdown, p, h1, h2, h3, span { color: black !important; }
        
        /* 2. 徹底消除輸入框右側的陰影與淡淡格線 */
        div[data-baseweb="input"], 
        div[data-baseweb="input"] > div,
        div[data-baseweb="input"] input {
            background-color: white !important;
            border-color: #dcdcdc !important; /* 設定一個淺灰色的統一邊框 */
            box-shadow: none !important;      /* 移除所有陰影 */
        }
        
        /* 針對日期選取器內部的特殊容器進行修正 */
        div[role="combobox"] {
            background-color: white !important;
            border: none !important;
        }

        /* 3. 強制按鈕內部的所有文字元素變白 */
        div.stButton > button {
            background-color: #000000 !important;
            border: 1px solid #000000 !important;
            font-weight: bold !important;
        }
        div.stButton > button * {
            color: #FFFFFF !important;
        }
        
        div.stButton > button:hover {
            background-color: #333333 !important;
        }

        /* 4. 側邊欄與輸入框整體調整 */
        [data-testid="stSidebar"] { border-right: 1px solid #f0f2f6; }
        input { 
            color: black !important; 
            background-color: white !important; 
        }
        </style>
        """, unsafe_allow_html=True)

# 定義開始計算按鈕
calculate_btn = st.sidebar.button("開始計算")

# --- 4. 主要邏輯判斷 ---
st.write(f"## 📈 David 乖離率線性回歸")

if not calculate_btn:
    # 初始提示訊息
    st.info("💡 請點開左上角選單 [ > ] 設定參數後按「開始計算」。")
else:
    # 抓取資料
    search_id = f"{stock_id}.TW" if stock_id.isdigit() else stock_id
    data = yf.download(search_id, start=start_date, end=end_date, auto_adjust=True)
    
    if not data.empty:
        # 取得公司名稱 (Ticker 物件)
        ticker_info = yf.Ticker(search_id)
        long_name = ticker_info.info.get('longName', search_id)
        st.write(f"### {search_id} - {long_name}")

        # --- 1. 處理 yfinance 可能產生的多層索引 (MultiIndex) ---
        # 如果欄位是多層的（例如包含 Ticker 名稱），則只取最內層的 Open, High, Low, Close
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 重設索引，將 Date 變成一個普通的欄位
        df = data.reset_index()

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
        # 使用剛建立的 Close_1D 確保欄位名稱正確
        df['MA'] = df['Close_1D'].rolling(window=ma_period).mean()

        # A. 計算移動平均線 (MA)
        df['MA'] = df['Close'].rolling(window=ma_period).mean()

        # B. 定義乖離率 (Bias Ratio)
        # 公式：(收盤價 / MA - 1) * 100
        df['Bias'] = ((df['Close'] / df['MA']) - 1) * 100
        
        # A. 計算移動平均線 (MA)
        # 使用扁平化後的 Close_1D 確保計算準確
        df['MA'] = df['Close_1D'].rolling(window=ma_period).mean()

        # B. 定義乖離率 (Bias Ratio)
        df['Bias'] = ((df['Close_1D'] / df['MA']) - 1) * 100
        
        # --- 關鍵修正：同時處理 NaN 與 Inf (無限大) ---
        # 1. 將無限大替換為 NaN 2. 刪除所有 NaN
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Bias']).reset_index(drop=True)

        # 增加防錯機制：如果過濾後資料太少，則不進行回歸
        if len(df) < 10:
            st.error(f"❌ 目前日期範圍內的有效資料太少（少於 10 筆），無法進行 {ma_period} 天回歸分析。請加長起始日期。")
        else:
            # C. 線性回歸計算
            X = np.array(df.index).reshape(-1, 1)
            Y = df['Bias'].values.reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X, Y) # 現在這裡不會再噴 NaN 錯誤了！
            
            df['Bias_Reg'] = model.predict(X)
    
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
        df['Bias_P2SD'] = df['Bias_Reg'] + (2 * sd_val) # 極端樂觀 (+2SD)
        df['Bias_P1SD'] = df['Bias_Reg'] + sd_val       # 樂觀 (+1SD)
        df['Bias_M1SD'] = df['Bias_Reg'] - sd_val       # 悲觀 (-1SD)
        df['Bias_M2SD'] = df['Bias_Reg'] - (2 * sd_val) # 極端悲觀 (-2SD)

        # F. 繪圖：使用 Plotly
        fig = go.Figure()

        # 1. 實際乖離率曲線 (主線) - 修改 x 為 df['Date_Str']
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias'], name='實際乖離率', line=dict(color='#17BECF', width=2)))

        # 2. 線性回歸線 (中心線) - 修改 x 為 df['Date_Str']
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_Reg'], name='回歸中線', line=dict(color='orange', dash='dash')))

        # 3. 標準差軌道線 - 全部修改 x 為 df['Date_Str']
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_P2SD'], name='+2SD 極端樂觀', line=dict(color='red', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_P1SD'], name='+1SD 樂觀', line=dict(color='pink', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_M1SD'], name='-1SD 悲觀', line=dict(color='lightgreen', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Bias_M2SD'], name='-2SD 極端悲觀', line=dict(color='green', width=1, dash='dot')))

        # 圖表佈局設定
        fig.update_layout(
            height=600,
            template=chart_template,
            hovermode="x unified",
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            # 全域字體設定：加深顏色
            font=dict(color=font_color, size=14), 
            
            xaxis=dict(
                type='category', 
                color=font_color,
                # X 軸刻度字體加深加粗
                tickfont=dict(color=font_color, size=12), 
                title=dict(text="日期", font=dict(color=font_color, size=14)),
                nticks=8,
                showgrid=False,   # 消除垂直格線
                zeroline=False    # 消除零軸線
            ),
            
            yaxis=dict(
                color=font_color,
                # Y 軸刻度字體加深加粗
                tickfont=dict(color=font_color, size=12), 
                title=dict(text="乖離率 (%)", font=dict(color=font_color, size=14)),
                showgrid=False,   # 消除水平格線
                zeroline=False    # 消除零軸線
            ),
            
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5,
                # 圖例字體加深
                font=dict(color=font_color, size=12) 
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # G. 顯示最後數據數據摘要
        st.markdown("---")
        st.subheader("📊 最後交易日數據摘要")
        last_row = df.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最後收盤價", f"{last_row['Close']:.2f}")
        col2.metric("目前乖離率", f"{last_row['Bias']:.2f}%")
        col3.metric("回歸中線值", f"{last_row['Bias_Reg']:.2f}%")
        col4.metric("標準差 (SD)", f"{sd_val:.2f}%")

    else:
        st.error("找不到股票資料，請檢查代號或日期設定。")