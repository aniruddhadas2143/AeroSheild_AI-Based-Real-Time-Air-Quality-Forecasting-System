# AI-Based-Real-Time-Air-Quality-Forecasting-System
Explainable Hybrid AI forecasts Air Quality Indices (AQI) by fusing XGBoost  and a CNN-BiLSTM sequential network. Features  scale via Min-Max before passing to a 1D CNN for spatial maps, Max Pooling for peak shifts, and a BiLSTM for 24h trends. A cached `@st.cache_resource` Streamlit UI outputs millisecond color-coded risk alerts.
