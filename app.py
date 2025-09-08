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
    color: #2c3e50;
    font-weight: 500;
}
.feature-negative {
    border-left-color: #dc3545;
    background-color: #fdf2f2;
}
.feature-positive {
    background-color: #f0f9f4;
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
.risk-increasing {
    background-color: #fdf2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #dc3545;
    padding: 12px;
    margin: 8px 0;
    border-radius: 6px;
    color: #721c24;
}
.risk-decreasing {
    background-color: #f0f9f4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #28a745;
    padding: 12px;
    margin: 8px 0;
    border-radius: 6px;
    color: #14532d;
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
        
        # Ensure data is clean and has proper variance
        X_clean = X.copy()
        
        # Check for and handle problematic features
        for col in X_clean.columns:
            # Handle constant features (zero variance)
            if X_clean[col].std() == 0:
                # Add small noise to constant features
                X_clean[col] = X_clean[col] + np.random.normal(0, 1e-6, size=len(X_clean))
            
            # Handle features with very small variance
            elif X_clean[col].std() < 1e-10:
                X_clean[col] = X_clean[col] + np.random.normal(0, 1e-6, size=len(X_clean))
            
            # Handle infinite or very large values
            X_clean[col] = np.clip(X_clean[col], -1e10, 1e10)
            
            # Replace any remaining NaN values
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # Create LIME explainer with more robust parameters
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_clean.values,
            feature_names=feature_names,
            mode='regression',
            verbose=False,  # Reduce verbosity to avoid warnings
            random_state=42,
            discretize_continuous=True,  # This can help with numerical stability
            sample_around_instance=True  # More stable sampling
        )
        
        # Explain instance with error handling
        exp = explainer.explain_instance(
            X_clean.iloc[instance_idx].values,
            underlying_model.predict,
            num_features=min(10, len(feature_names)),  # Ensure we don't exceed available features
            num_samples=1000  # Reduce samples if needed for stability
        )
        
        return exp
    except Exception as e:
        # More specific error handling
        if "scale parameter must be positive" in str(e) or "truncnorm" in str(e):
            st.warning("⚠️ LIME explanation cannot be generated due to data distribution issues. This can happen when features have very low variance or numerical stability problems.")
            st.info("💡 Try selecting a different region/time period or check if your data has sufficient variance in the features.")
        else:
            st.warning(f"LIME explanation generation failed: {str(e)}")
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
        df = validate_csv(uploaded_file)
        
        if df is not None:
            st.success(f"Data validated successfully! Loaded {len(df)} rows with {len(df.columns)} columns.")
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
                st.write(f"Data shape: {df.shape}")
            
            X = df[MODEL_FEATURES].copy()
            
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    try:
                        if col == 'Laju_Inflasi':
                            X[col] = X[col].astype(str).str.replace(',', '.').astype(float)
                        else:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        st.error(f"Failed to convert column '{col}' to numeric")
                        return
            
            if 'predictions_made' not in st.session_state:
                st.session_state.predictions_made = False
            if 'predictions' not in st.session_state:
                st.session_state.predictions = None
            if 'df_with_predictions' not in st.session_state:
                st.session_state.df_with_predictions = None
            
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
                            predictions = model.predict(X)
                            df_with_predictions = df.copy()
                            df_with_predictions['Risk_Score'] = predictions
                            
                            st.session_state.predictions = predictions
                            st.session_state.df_with_predictions = df_with_predictions
                            st.session_state.predictions_made = True
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            return
                
                if st.session_state.predictions_made:
                    df_with_predictions = st.session_state.df_with_predictions
                    predictions = st.session_state.predictions
                    
                    st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)
                    
                    # --- PERBAIKAN: Hapus pembungkus HTML di sekitar st.metric ---
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Risk Score", f"{np.mean(predictions):.3f}")
                    with col2:
                        st.metric("Min Risk Score", f"{np.min(predictions):.3f}")
                    with col3:
                        st.metric("Max Risk Score", f"{np.max(predictions):.3f}")
                    
                    with st.expander("📊 Detailed Predictions Table", expanded=False):
                        st.dataframe(
                            df_with_predictions[['kabupaten_kota', 'tahun', 'kuartal', 'Risk_Score']].sort_values('Risk_Score', ascending=False),
                            use_container_width=True
                        )
                    
                    # ... (Sisa kode SHAP tetap sama karena sudah benar) ...
                    st.markdown('<div class="section-header">🔍 Explainable AI (XAI) Insights</div>', unsafe_allow_html=True)
                    st.markdown("#### 📊 Global Feature Importance (SHAP)")
                    st.markdown("*Understanding which features have the most impact on risk predictions across all instances*")

                    with st.expander("🔧 Filter Options for SHAP Analysis", expanded=False):
                        shap_filter_col1, shap_filter_col2, shap_filter_col3 = st.columns(3)

                        with shap_filter_col1:
                            shap_selected_region = st.selectbox("Filter by Kabupaten/Kota (optional)", options=["All"] + sorted(df_with_predictions['kabupaten_kota'].unique()), key="shap_region_filter")
                        with shap_filter_col2:
                            shap_selected_year = st.selectbox("Filter by Year (optional)", options=["All"] + sorted(df_with_predictions['tahun'].unique()), key="shap_year_filter")
                        with shap_filter_col3:
                            shap_selected_quarter = st.selectbox("Filter by Quarter (optional)", options=["All"] + sorted(df_with_predictions['kuartal'].unique()), key="shap_quarter_filter")

                    # Apply filters for SHAP analysis
                    shap_filtered_df = df_with_predictions.copy()
                    if shap_selected_region != "All":
                        shap_filtered_df = shap_filtered_df[shap_filtered_df['kabupaten_kota'] == shap_selected_region]
                    if shap_selected_year != "All":
                        shap_filtered_df = shap_filtered_df[shap_filtered_df['tahun'] == shap_selected_year]
                    if shap_selected_quarter != "All":
                        shap_filtered_df = shap_filtered_df[shap_filtered_df['kuartal'] == shap_selected_quarter]
                    
                    if not shap_filtered_df.empty:
                        X_filtered = shap_filtered_df[MODEL_FEATURES]
                        shap_values = generate_shap_explanations(model, X_filtered, MODEL_FEATURES)
                        if shap_values is not None:
                            filter_info_parts = []
                            if shap_selected_region != "All": filter_info_parts.append(f"Region: {shap_selected_region}")
                            if shap_selected_year != "All": filter_info_parts.append(f"Year: {shap_selected_year}")
                            if shap_selected_quarter != "All": filter_info_parts.append(f"Quarter: {shap_selected_quarter}")
                            filter_info = ', '.join(filter_info_parts)
                            
                            st.info(f"📍 SHAP analysis filtered by: {filter_info if filter_info else 'All Data'} ({len(X_filtered)} instances)")
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            shap.summary_plot(shap_values, X_filtered, show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.warning("No data available for the selected filters")

                    # ... (LIME section remains mostly the same, just showing the correct logic)
                    st.markdown("#### 🔎 Local Explanations (LIME)")
                    st.markdown("*Understanding individual predictions for specific regions and time periods*")
                    
                    with st.container():
                        st.markdown("**Select Instance for Detailed Analysis:**")
                        col1, col2, col3 = st.columns(3)
                        
                        # Set default selections for LIME
                        unique_regions = sorted(df_with_predictions['kabupaten_kota'].unique())
                        unique_years = sorted(df_with_predictions['tahun'].unique())
                        unique_quarters = sorted(df_with_predictions['kuartal'].unique())

                        if st.session_state.selected_region is None or st.session_state.selected_region not in unique_regions:
                            st.session_state.selected_region = unique_regions[0]
                        if st.session_state.selected_year is None or st.session_state.selected_year not in unique_years:
                            st.session_state.selected_year = unique_years[-1]
                        if st.session_state.selected_quarter is None or st.session_state.selected_quarter not in unique_quarters:
                            st.session_state.selected_quarter = unique_quarters[0]
                        
                        with col1:
                            st.session_state.selected_region = st.selectbox("Select Kabupaten/Kota", options=unique_regions, index=unique_regions.index(st.session_state.selected_region))
                        with col2:
                            st.session_state.selected_year = st.selectbox("Select Year", options=unique_years, index=unique_years.index(st.session_state.selected_year))
                        with col3:
                            st.session_state.selected_quarter = st.selectbox("Select Quarter", options=unique_quarters, index=unique_quarters.index(st.session_state.selected_quarter))
                    
                    filtered_df = df_with_predictions[
                        (df_with_predictions['kabupaten_kota'] == st.session_state.selected_region) & 
                        (df_with_predictions['tahun'] == st.session_state.selected_year) & 
                        (df_with_predictions['kuartal'] == st.session_state.selected_quarter)
                    ]
                    
                    if not filtered_df.empty:
                        instance_idx = filtered_df.index[0]
                        
                        # Get the actual values for this instance
                        instance_values = X.iloc[instance_idx]
                        predicted_risk = predictions[instance_idx]
                        
                        # Try LIME explanation first
                        lime_exp = generate_lime_explanation(model, X, instance_idx, MODEL_FEATURES)
                        
                        if lime_exp:
                            st.markdown(f"**📋 Feature Impact Analysis for {st.session_state.selected_region} - Q{st.session_state.selected_quarter} {st.session_state.selected_year}**")
                            st.markdown(f"*Predicted Risk Score: {predicted_risk:.3f}*")
                            
                            exp_list = lime_exp.as_list()
                            positive_features = sorted([(f, w) for f, w in exp_list if w > 0], key=lambda item: item[1], reverse=True)
                            negative_features = sorted([(f, w) for f, w in exp_list if w < 0], key=lambda item: item[1])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if positive_features:
                                    st.markdown("**🔴 Risk Increasing Factors:**")
                                    for feature, weight in positive_features:
                                        st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Impact: <strong>+{weight:.3f}</strong></div>', unsafe_allow_html=True)
                            with col2:
                                if negative_features:
                                    st.markdown("**🟢 Risk Decreasing Factors:**")
                                    for feature, weight in negative_features:
                                        st.markdown(f'<div class="risk-decreasing"><strong>{feature}</strong><br/>Impact: <strong>{weight:.3f}</strong></div>', unsafe_allow_html=True)
                        else:
                            # Fallback: Show feature values and basic statistics when LIME fails
                            st.markdown(f"**📋 Feature Values Analysis for {st.session_state.selected_region} - Q{st.session_state.selected_quarter} {st.session_state.selected_year}**")
                            st.markdown(f"*Predicted Risk Score: {predicted_risk:.3f}*")
                            st.info("📊 Showing feature values and comparisons with dataset averages:")
                            
                            # Calculate dataset statistics
                            feature_means = X.mean()
                            feature_stds = X.std()
                            
                            # Create comparison analysis
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📈 Above Average Features:**")
                                above_avg_features = []
                                for feature in MODEL_FEATURES:
                                    if instance_values[feature] > feature_means[feature]:
                                        deviation = (instance_values[feature] - feature_means[feature]) / feature_stds[feature]
                                        above_avg_features.append((feature, deviation, instance_values[feature]))
                                
                                above_avg_features.sort(key=lambda x: x[1], reverse=True)
                                for feature, deviation, value in above_avg_features[:5]:
                                    st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Value: <strong>{value:.3f}</strong><br/>Std Dev above mean: <strong>+{deviation:.2f}σ</strong></div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("**📉 Below Average Features:**")
                                below_avg_features = []
                                for feature in MODEL_FEATURES:
                                    if instance_values[feature] < feature_means[feature]:
                                        deviation = (feature_means[feature] - instance_values[feature]) / feature_stds[feature]
                                        below_avg_features.append((feature, deviation, instance_values[feature]))
                                
                                below_avg_features.sort(key=lambda x: x[1], reverse=True)
                                for feature, deviation, value in below_avg_features[:5]:
                                    st.markdown(f'<div class="risk-decreasing"><strong>{feature}</strong><br/>Value: <strong>{value:.3f}</strong><br/>Std Dev below mean: <strong>-{deviation:.2f}σ</strong></div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"No data found for {st.session_state.selected_region} in Q{st.session_state.selected_quarter} {st.session_state.selected_year}")
                    
                    # ... (Sisa kode sama) ...
                    st.markdown("#### 📈 Feature Distribution Analysis")
                    st.markdown("*Explore the distribution of key socioeconomic indicators*")
                    
                    col1, col2 = st.columns([2, 1])
                    with col2:
                        selected_feature = st.selectbox("Select feature to visualize:", MODEL_FEATURES, help="Choose a feature to see its distribution across all data points")
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
                            
                    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
                    csv = df_with_predictions.to_csv(index=False).encode('utf-8')
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
        # --- PERBAIKAN: Gabungkan HTML ke dalam satu st.markdown ---
        st.markdown("""
        <div class="info-card">
            <h3>👋 Welcome to the Socioeconomic Risk Predictor!</h3>
            <p><em>Advanced machine learning analytics for socioeconomic risk assessment</em></p>
        </div>
        """, unsafe_allow_html=True)
        
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
            Your CSV file should include columns like `kabupaten_kota`, `tahun`, `kuartal`, `PDRB`, `Upah Minimum`, `Indeks_Pembangunan_Manusia`, etc. 
            For a full list of the 16 required numeric features, please refer to the model documentation.
            """)
        
        with col2:
            st.markdown("### 🔬 AI Model Features")
            st.info("""
            **🎯 Prediction Capabilities:**
            - Risk score calculation
            - Multi-factor analysis
            
            **🔍 Explainable AI:**
            - SHAP global importance
            - LIME local explanations
            
            **📊 Visualizations:**
            - Interactive charts
            - Distribution plots
            
            **💾 Export Options:**
            - Complete results
            - Detailed predictions
            """)
            
            st.markdown("### 🆘 Support")
            st.markdown("""
            Need help? Check our documentation or contact support for:
            - Data formatting assistance
            - Model interpretation
            - Technical troubleshooting
            """)

if __name__ == "__main__":
    main()
