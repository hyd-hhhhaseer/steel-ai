import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import os

# 1. 页面设置
st.set_page_config(page_title="材料工程AI平台", layout="wide")

# 2. 智能读取数据 (自动处理编码问题)
@st.cache_resource
def load_data():
    file_path = "data.csv" # 咱们统一好的文件名
    if not os.path.exists(file_path):
        return None, "⚠️ 找不到 data.csv 文件，请检查GitHub是否上传正确。"
    
    # 尝试两种常见的中文编码
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            return None, "❌ 文件编码读取失败，请尝试另存为标准CSV格式。"
            
    # 数据清洗：填补空值，确保数值列是数字
    num_cols = ['C_Avg', 'Cr_Avg', 'Mn_Avg', 'Mo_Avg', 'Ni_Avg', 'V_Avg', 'HRC_Avg']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df, "✅ 数据加载成功"

df, msg = load_data()

# 3. 如果没数据，显示报错；有数据则运行模型
if df is None:
    st.error(msg)
else:
    # 训练模型
    X = df[['C_Avg', 'Cr_Avg', 'Mn_Avg', 'Mo_Avg', 'Ni_Avg', 'V_Avg']]
    y = df['HRC_Avg']
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3).fit(X, y)

    # --- 界面开始 ---
    st.title("🔩 材料工程 AI 助手")
    
    # 侧边栏：输入成分
    st.sidebar.header("🧪 成分配比 (%)")
    def get_input():
        c = st.sidebar.slider('C (碳)', 0.0, 3.0, 0.4)
        cr = st.sidebar.slider('Cr (铬)', 0.0, 20.0, 12.0)
        mn = st.sidebar.slider('Mn (锰)', 0.0, 5.0, 0.5)
        mo = st.sidebar.slider('Mo (钼)', 0.0, 5.0, 0.5)
        ni = st.sidebar.slider('Ni (镍)', 0.0, 5.0, 0.0)
        v = st.sidebar.slider('V (钒)', 0.0, 5.0, 0.0)
        return pd.DataFrame([[c, cr, mn, mo, ni, v]], columns=X.columns)
    
    input_data = get_input()
    
    # 主界面分两栏
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 性能预测")
        pred = model.predict(input_data)[0]
        st.metric("预测硬度 (HRC)", f"{pred:.1f}")
        
        if pred > 58: st.warning("高硬度：适用于冷作模具/刀具")
        elif pred > 40: st.info("中硬度：适用于热作/塑料模具")
        else: st.success("低硬度：韧性较好或预硬钢")

    with col2:
        st.subheader("📊 元素影响分析")
        importance = pd.DataFrame({'元素': X.columns, '重要性': model.feature_importances_})
        st.plotly_chart(px.bar(importance, x='元素', y='重要性'), use_container_width=True)

    st.divider()
    
    # 搜索功能
    st.subheader("🔍 牌号/标准检索")
    keyword = st.text_input("输入关键词（如：2083, 耐腐蚀, GB）：")
    if keyword:
        # 模糊搜索所有文本列
        mask = df.astype(str).apply(lambda x: x.str.contains(keyword, case=False)).any(axis=1)
        res = df[mask]
        st.write(f"找到 {len(res)} 条结果：")
        st.dataframe(res, hide_index=True)
