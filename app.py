import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Socioeconomic Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 600;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: 500;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}
.explanation-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border-left: 4px solid #1f77b4;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e1e8ed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
.feature-importance {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 3px solid #28a745;
}
.feature-negative {
    border-left-color: #dc3545;
}
.sidebar-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 1rem;
}
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Expected columns from the training data (model was trained on these 16 numeric features)
MODEL_FEATURES = [
    'Garis_Kemiskinan', 'Indeks_Pembangunan_Manusia', 'Persen_Penduduk_Miskin',
    'Tingkat Pengangguran Terbuka', 'Upah Minimum', 'Jumlah Penduduk (Ribu)',
    'Laju Pertumbuhan Penduduk per Tahun', 'Persentase Penduduk',
    'Kepadatan Penduduk per km persegi (Km2)', 'Rasio Jenis Kelamin Penduduk',
    'PDRB', 'Laju_Inflasi', 'Gini_Ratio', 'investasi_per_kapita',
    'Jumlah Perusahaan Kecil', 'Jumlah Perusahaan'
]

# Full expected columns in the original data (for validation)
FULL_EXPECTED_COLUMNS = MODEL_FEATURES + [
    'kabupaten_kota', 'tahun', 'kuartal', 'Proksi Inflasi'
]

def load_model():
    """Load the pre-trained prediction pipeline"""
    try:
        pipeline = joblib.load('Pipeline/risk_prediction_pipeline.pkl')
        
        # Check if pipeline has feature names stored
        if hasattr(pipeline, 'feature_names_in_'):
            st.info(f"Model trained on features: {list(pipeline.feature_names_in_)}")
        
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def validate_csv(uploaded_file):
    """Validate uploaded CSV file structure"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if all expected columns are present
        missing_cols = set(FULL_EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.warning(f"Missing columns: {', '.join(missing_cols)}")
            return None
        
        # Check that all model features are numeric or can be converted
        for col in MODEL_FEATURES:
            if col in df.columns:
                # Try to convert to numeric if not already
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        # Special handling for Laju_Inflasi which might contain comma decimals
                        if col == 'Laju_Inflasi':
                            # Replace commas with dots and convert to numeric
                            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                        else:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Check if conversion resulted in too many NaN values
                        if df[col].isna().sum() > len(df) * 0.1:  # More than 10% NaN
                            st.warning(f"Column '{col}' contains non-numeric values that cannot be converted")
                            return None
                    except Exception as e:
                        st.warning(f"Column '{col}' should be numeric. Conversion error: {e}")
                        return None
        
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def generate_shap_explanations(model, X, feature_names):
    """Generate SHAP explanations for the model"""
    try:
        # Create explainer
        explainer = shap.Explainer(model.named_steps['xgb_model'] if hasattr(model, 'named_steps') else model)
        
        # Get SHAP values
        shap_values = explainer(X)
        
        return shap_values
    except Exception as e:
        st.warning(f"SHAP explanation generation failed: {e}")
        return None

def generate_lime_explanation(model, X, instance_idx, feature_names):
    """Generate LIME explanation for a specific instance"""
    try:
        # Get the underlying model from pipeline if needed
        if hasattr(model, 'named_steps'):
            underlying_model = model.named_steps['xgb_model']
        else:
            underlying_model = model
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=feature_names,
            mode='regression',
            verbose=True,
            random_state=42
        )
        
        # Explain instance
        exp = explainer.explain_instance(
            X.iloc[instance_idx].values,
            underlying_model.predict,
            num_features=10
        )
        
        return exp
    except Exception as e:
        st.warning(f"LIME explanation generation failed: {e}")
        return None

def main():
    st.markdown('<div class="main-header">📊 Socioeconomic Risk Predictor with XAI</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load prediction model. Please ensure the pipeline file exists.")
        return
    
    # Initialize session state for filter persistence
    if 'shap_selected_region' not in st.session_state:
        st.session_state.shap_selected_region = "All"
    if 'shap_selected_year' not in st.session_state:
        st.session_state.shap_selected_year = "All"
    if 'shap_selected_quarter' not in st.session_state:
        st.session_state.shap_selected_quarter = "All"
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = None
    if 'selected_quarter' not in st.session_state:
        st.session_state.selected_quarter = None
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📁 Data Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV file with socioeconomic data",
            type=['csv'],
            help="Upload a CSV file with the expected column structure"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
            
        st.markdown("---")
        st.markdown('<div class="sidebar-header">📊 Model Information</div>', unsafe_allow_html=True)
        st.markdown("""
        **Expected Features:**
        - Economic indicators
        - Social indicators  
        - Demographic data
        - Regional identifiers
        - Time series data
        """)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">🔍 Analysis Features</div>', unsafe_allow_html=True)
        st.markdown("""
        - **SHAP Analysis**: Global feature importance
        - **LIME Explanations**: Local interpretability
        - **Interactive Visualizations**: Data insights
        - **Export Results**: Download predictions
        """)
    
    # Main content area
    if uploaded_file:
        # Validate and load data
        df = validate_csv(uploaded_file)
        
        if df is not None:
            st.success(f"Data validated successfully! Loaded {len(df)} rows with {len(df.columns)} columns.")
            
            # Display data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
                st.write(f"Data shape: {df.shape}")
            
            # Prepare features for prediction (only the 16 numeric features the model was trained on)
            X = df[MODEL_FEATURES].copy()
            
            # Ensure all features are numeric (handle any remaining conversion issues)
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    try:
                        # Special handling for Laju_Inflasi which might contain comma decimals
                        if col == 'Laju_Inflasi':
                            X[col] = X[col].astype(str).str.replace(',', '.').astype(float)
                        else:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        st.error(f"Failed to convert column '{col}' to numeric")
                        return
            
            # Initialize session state for predictions
            if 'predictions_made' not in st.session_state:
                st.session_state.predictions_made = False
            if 'predictions' not in st.session_state:
                st.session_state.predictions = None
            if 'df_with_predictions' not in st.session_state:
                st.session_state.df_with_predictions = None
            
            # Make predictions button
            col1, col2 = st.columns([3, 1])
            with col1:
                make_predictions = st.button("🚀 Make Predictions", type="primary")
            with col2:
                if st.session_state.predictions_made and st.button("🔄 Reset Predictions"):
                    st.session_state.predictions_made = False
                    st.session_state.predictions = None
                    st.session_state.df_with_predictions = None
                    st.rerun()
            
            if make_predictions or st.session_state.predictions_made:
                if not st.session_state.predictions_made:
                    with st.spinner("Making predictions and generating explanations..."):
                        try:
                            # Predict
                            predictions = model.predict(X)
                            
                            # Add predictions to dataframe
                            df_with_predictions = df.copy()
                            df_with_predictions['Risk_Score'] = predictions
                            
                            # Store in session state
                            st.session_state.predictions = predictions
                            st.session_state.df_with_predictions = df_with_predictions
                            st.session_state.predictions_made = True
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            return
                
                # Display results from session state
                if st.session_state.predictions_made:
                    df_with_predictions = st.session_state.df_with_predictions
                    predictions = st.session_state.predictions
                    
                    # Display results
                    st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)
                    
                    # Results overview
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Average Risk Score", f"{np.mean(predictions):.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Min Risk Score", f"{np.min(predictions):.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Max Risk Score", f"{np.max(predictions):.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display predictions with original data
                    with st.expander("📊 Detailed Predictions Table", expanded=False):
                        st.dataframe(
                            df_with_predictions[['kabupaten_kota', 'tahun', 'kuartal', 'Risk_Score']].sort_values('Risk_Score', ascending=False),
                            use_container_width=True
                        )
                    
                    # XAI Section
                    st.markdown('<div class="section-header">🔍 Explainable AI (XAI) Insights</div>', unsafe_allow_html=True)
                    
                    # SHAP Global Explanations
                    st.markdown("#### 📊 Global Feature Importance (SHAP)")
                    st.markdown("*Understanding which features have the most impact on risk predictions across all instances*")
                    
                    # Add filter options for SHAP analysis
                    with st.expander("🔧 Filter Options for SHAP Analysis", expanded=False):
                        shap_filter_col1, shap_filter_col2, shap_filter_col3 = st.columns(3)
                    
                    with shap_filter_col1:
                        shap_selected_region = st.selectbox(
                            "Filter by Kabupaten/Kota (optional)",
                            options=["All"] + sorted(df_with_predictions['kabupaten_kota'].unique()),
                            index=0 if st.session_state.shap_selected_region == "All" else (["All"] + sorted(df_with_predictions['kabupaten_kota'].unique())).index(st.session_state.shap_selected_region),
                            key="shap_region_filter"
                        )
                        st.session_state.shap_selected_region = shap_selected_region
                    
                    with shap_filter_col2:
                        shap_selected_year = st.selectbox(
                            "Filter by Year (optional)",
                            options=["All"] + sorted(df_with_predictions['tahun'].unique()),
                            index=0 if st.session_state.shap_selected_year == "All" else (["All"] + sorted(df_with_predictions['tahun'].unique())).index(st.session_state.shap_selected_year),
                            key="shap_year_filter"
                        )
                        st.session_state.shap_selected_year = shap_selected_year
                    
                    with shap_filter_col3:
                        shap_selected_quarter = st.selectbox(
                            "Filter by Quarter (optional)",
                            options=["All"] + sorted(df_with_predictions['kuartal'].unique()),
                            index=0 if st.session_state.shap_selected_quarter == "All" else (["All"] + sorted(df_with_predictions['kuartal'].unique())).index(st.session_state.shap_selected_quarter),
                            key="shap_quarter_filter"
                        )
                        st.session_state.shap_selected_quarter = shap_selected_quarter
                    
                    # Apply filters for SHAP analysis
                    shap_filtered_df = df_with_predictions.copy()
                    if shap_selected_region != "All":
                        shap_filtered_df = shap_filtered_df[shap_filtered_df['kabupaten_kota'] == shap_selected_region]
                    if shap_selected_year != "All":
                        shap_filtered_df = shap_filtered_df[shap_filtered_df['tahun'] == shap_selected_year]
                    if shap_selected_quarter != "All":
                        shap_filtered_df = shap_filtered_df[shap_filtered_df['kuartal'] == shap_selected_quarter]
                    
                    if not shap_filtered_df.empty:
                        X_filtered = shap_filtered_df[MODEL_FEATURES].copy()
                        
                        # Generate SHAP values for filtered data
                        shap_values = generate_shap_explanations(model, X_filtered, MODEL_FEATURES)
                        
                        if shap_values is not None:
                            # Display filter info
                            filter_info = []
                            if shap_selected_region != "All":
                                filter_info.append(f"Region: {shap_selected_region}")
                            if shap_selected_year != "All":
                                filter_info.append(f"Year: {shap_selected_year}")
                            if shap_selected_quarter != "All":
                                filter_info.append(f"Quarter: {shap_selected_quarter}")
                            
                            if filter_info:
                                st.info(f"📍 SHAP analysis filtered by: {', '.join(filter_info)} ({len(X_filtered)} instances)")
                            else:
                                st.info(f"📍 SHAP analysis for all data ({len(X_filtered)} instances)")
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            shap.summary_plot(shap_values, X_filtered, show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.warning("No data available for the selected filters")
                    
                    # LIME Local Explanations
                    st.markdown("#### 🔎 Local Explanations (LIME)")
                    st.markdown("*Understanding individual predictions for specific regions and time periods*")
                    
                    # Create selection dropdowns for specific region, year, and quarter
                    with st.container():
                        st.markdown("**Select Instance for Detailed Analysis:**")
                        col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Set default region if not set
                        if st.session_state.selected_region is None:
                            st.session_state.selected_region = sorted(df_with_predictions['kabupaten_kota'].unique())[0]
                        
                        selected_region = st.selectbox(
                            "Select Kabupaten/Kota",
                            options=sorted(df_with_predictions['kabupaten_kota'].unique()),
                            index=sorted(df_with_predictions['kabupaten_kota'].unique()).index(st.session_state.selected_region),
                            key="lime_region_select"
                        )
                        st.session_state.selected_region = selected_region
                    
                    with col2:
                        # Set default year if not set
                        if st.session_state.selected_year is None:
                            st.session_state.selected_year = sorted(df_with_predictions['tahun'].unique())[-1]  # Latest year
                        
                        selected_year = st.selectbox(
                            "Select Year",
                            options=sorted(df_with_predictions['tahun'].unique()),
                            index=sorted(df_with_predictions['tahun'].unique()).index(st.session_state.selected_year),
                            key="lime_year_select"
                        )
                        st.session_state.selected_year = selected_year
                    
                    with col3:
                        # Set default quarter if not set
                        if st.session_state.selected_quarter is None:
                            st.session_state.selected_quarter = sorted(df_with_predictions['kuartal'].unique())[0]
                        
                        selected_quarter = st.selectbox(
                            "Select Quarter",
                            options=sorted(df_with_predictions['kuartal'].unique()),
                            index=sorted(df_with_predictions['kuartal'].unique()).index(st.session_state.selected_quarter),
                            key="lime_quarter_select"
                        )
                        st.session_state.selected_quarter = selected_quarter
                    
                    # Find the specific instance based on selection
                    filtered_df = df_with_predictions[
                        (df_with_predictions['kabupaten_kota'] == selected_region) & 
                        (df_with_predictions['tahun'] == selected_year) & 
                        (df_with_predictions['kuartal'] == selected_quarter)
                    ]
                    
                    if not filtered_df.empty:
                        instance_idx = filtered_df.index[0]
                        
                        lime_exp = generate_lime_explanation(model, X, instance_idx, MODEL_FEATURES)
                        if lime_exp:
                            # Display LIME explanation
                            st.markdown("**📋 Feature Impact Analysis**")
                            st.markdown(f"*Analysis for {selected_region} - Q{selected_quarter} {selected_year}*")
                            
                            # Create a more professional display for LIME results
                            exp_list = lime_exp.as_list()
                            
                            # Separate positive and negative contributions
                            positive_features = [(f, w) for f, w in exp_list if w > 0]
                            negative_features = [(f, w) for f, w in exp_list if w < 0]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if positive_features:
                                    st.markdown("**🔴 Risk Increasing Factors:**")
                                    for feature, weight in positive_features:
                                        st.markdown(f'<div class="feature-importance">{feature}: <strong>+{weight:.3f}</strong></div>', unsafe_allow_html=True)
                            
                            with col2:
                                if negative_features:
                                    st.markdown("**🟢 Risk Decreasing Factors:**")
                                    for feature, weight in negative_features:
                                        st.markdown(f'<div class="feature-importance feature-negative">{feature}: <strong>{weight:.3f}</strong></div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"No data found for {selected_region} in Q{selected_quarter} {selected_year}")
                    
                    # Feature distributions
                    st.markdown("#### 📈 Feature Distribution Analysis")
                    st.markdown("*Explore the distribution of key socioeconomic indicators*")
                    
                    col1, col2 = st.columns([2, 1])
                    with col2:
                        selected_feature = st.selectbox(
                            "Select feature to visualize:", 
                            MODEL_FEATURES, 
                            help="Choose a feature to see its distribution across all data points"
                        )
                    
                    with col1:
                        if selected_feature in X.columns:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(X[selected_feature], kde=True, ax=ax, color='#3498db', alpha=0.7)
                            ax.set_title(f"Distribution of {selected_feature}", fontsize=14, fontweight='bold')
                            ax.set_xlabel(selected_feature, fontsize=12)
                            ax.set_ylabel("Frequency", fontsize=12)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    
                    # Download results
                    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
                    csv = df_with_predictions.to_csv(index=False)
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📥 Download Complete Results with Predictions",
                            data=csv,
                            file_name="socioeconomic_risk_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
    else:
        # Welcome message and instructions
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 👋 Welcome to the Socioeconomic Risk Predictor!")
        st.markdown("*Advanced machine learning analytics for socioeconomic risk assessment*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            **Follow these simple steps to analyze your data:**
            
            1. **📤 Upload Data**: Use the sidebar to upload your CSV file with socioeconomic indicators
            2. **⚡ Generate Predictions**: Click the prediction button to run the AI model
            3. **🔍 Explore Insights**: Analyze results with interactive explanations
            4. **📊 Download Results**: Export your findings for further analysis
            """)
            
            st.markdown("### 📋 Required Data Structure")
            st.markdown("""
            Your CSV file should include these key components:
            
            **🏛️ Regional Data:**
            - `kabupaten_kota` - Region identifiers
            - `nama_kabupaten_kota` - Region names
            
            **💰 Economic Indicators:**
            - `PDRB` - Gross Regional Domestic Product
            - `Upah Minimum` - Minimum wage
            - `investasi_per_kapita` - Investment per capita
            - `Laju_Inflasi` - Inflation rate
            
            **👥 Social Indicators:**
            - `Indeks_Pembangunan_Manusia` - Human Development Index
            - `Persen_Penduduk_Miskin` - Poverty percentage
            - `Tingkat Pengangguran Terbuka` - Unemployment rate
            
            **📈 Demographic Data:**
            - `Jumlah Penduduk` - Population count
            - `Laju Pertumbuhan Penduduk` - Population growth rate
            - `Kepadatan Penduduk` - Population density
            
            **📅 Time Series:**
            - `tahun` - Year
            - `kuartal` - Quarter
            """)
        
        with col2:
            st.markdown("### 🔬 AI Model Features")
            st.info("""
            **🎯 Prediction Capabilities:**
            - Risk score calculation
            - Multi-factor analysis
            - Time series patterns
            
            **🔍 Explainable AI:**
            - SHAP global importance
            - LIME local explanations
            - Feature impact analysis
            
            **📊 Visualizations:**
            - Interactive charts
            - Distribution plots
            - Trend analysis
            
            **💾 Export Options:**
            - Complete results
            - Detailed predictions
            - Analysis reports
            """)
            
            st.markdown("### � Support")
            st.markdown("""
            Need help? Check our documentation or contact support for:
            - Data formatting assistance
            - Model interpretation
            - Technical troubleshooting
            """)

if __name__ == "__main__":
    main()
