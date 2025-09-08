import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import openai  # --- MODIFIKASI: Menggunakan library openai ---
import io

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Socioeconomic Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (tidak ada perubahan)
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
.sidebar-header {
    font-size: 1.2rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1rem;
    text-align: center;
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

# --- Variabel Global & Fungsi Bawaan (tidak ada perubahan) ---
MODEL_FEATURES = [
    'Garis_Kemiskinan', 'Indeks_Pembangunan_Manusia', 'Persen_Penduduk_Miskin',
    'Tingkat Pengangguran Terbuka', 'Upah Minimum', 'Jumlah Penduduk (Ribu)',
    'Laju Pertumbuhan Penduduk per Tahun', 'Persentase Penduduk',
    'Kepadatan Penduduk per km persegi (Km2)', 'Rasio Jenis Kelamin Penduduk',
    'PDRB', 'Laju_Inflasi', 'Gini_Ratio', 'investasi_per_kapita',
    'Jumlah Perusahaan Kecil', 'Jumlah Perusahaan'
]
FULL_EXPECTED_COLUMNS = MODEL_FEATURES + ['kabupaten_kota', 'tahun', 'kuartal', 'Proksi Inflasi']

def load_model():
    try:
        pipeline = joblib.load('Pipeline/risk_prediction_pipeline.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def validate_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = set(FULL_EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.warning(f"Missing columns: {', '.join(missing_cols)}")
            return None
        for col in MODEL_FEATURES:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

def generate_shap_explanations(model, X, feature_names):
    """Generate SHAP explanations for the model"""
    try:
        # Create explainer with better error handling
        if hasattr(model, 'named_steps'):
            # Pipeline model
            underlying_model = model.named_steps['xgb_model']
        else:
            underlying_model = model
        
        # Try different explainer types based on model type
        try:
            # For tree-based models, use TreeExplainer (more efficient)
            if hasattr(underlying_model, 'get_booster') or 'XGB' in str(type(underlying_model)):
                explainer = shap.TreeExplainer(underlying_model)
            else:
                explainer = shap.Explainer(underlying_model)
        except Exception:
            # Fallback to general Explainer
            explainer = shap.Explainer(underlying_model)
        
        # Get SHAP values with error handling
        shap_values = explainer(X)
        
        return shap_values
    except Exception as e:
        st.warning(f"SHAP explanation generation failed: {e}")
        st.info("💡 Tip: Pastikan model kompatibel dengan SHAP dan data tidak memiliki nilai yang bermasalah.")
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

# --- TAMBAHAN: Fungsi untuk memanggil DeepSeek API ---
def generate_narrative_explanation(client, region, year, quarter, risk_score, positive_features, negative_features, shap_analysis=None):
    """Generate a narrative explanation using DeepSeek API via openai client."""
    if not client:
        return "Layanan AI tidak dikonfigurasi."

    pos_factors_str = "\n".join([f"- `{factor}` (Kontribusi LIME: {weight:.3f})" for factor, weight in positive_features])
    neg_factors_str = "\n".join([f"- `{factor}` (Kontribusi LIME: {weight:.3f})" for factor, weight in negative_features])
    
    # Add SHAP analysis if available
    shap_section = ""
    if shap_analysis:
        shap_pos_str = "\n".join([f"- `{factor}` (Rata-rata SHAP: {weight:.4f})" for factor, weight in shap_analysis['positive_factors'][:5]])
        shap_neg_str = "\n".join([f"- `{factor}` (Rata-rata SHAP: {weight:.4f})" for factor, weight in shap_analysis['negative_factors'][:5]])
        
        shap_section = f"""
    **Analisis Global (SHAP) - Pola Umum Model:**
    **Faktor Global Peningkat Risiko:**
    {shap_pos_str}
    **Faktor Global Penurun Risiko:**
    {shap_neg_str}
    """
    
    prompt_content = f"""
    Anda adalah seorang analis ekonomi ahli yang ditugaskan untuk menjelaskan prediksi risiko sosioekonomi untuk seorang kepala daerah atau pembuat kebijakan.

    **Konteks Analisis:**
    - **Wilayah:** {region}
    - **Periode:** Kuartal {quarter}, Tahun {year}
    - **Skor Risiko yang Diprediksi Model:** {risk_score:.3f} (skor lebih tinggi berarti risiko lebih besar)

    **Faktor-faktor Kunci dari Model (berdasarkan LIME - Analisis Lokal untuk Wilayah Ini):**
    **Faktor Peningkat Risiko Teratas:**
    {pos_factors_str}
    **Faktor Penurun Risiko Teratas:**
    {neg_factors_str}
    {shap_section}

    **Tugas Anda:**
    Tulis laporan analisis yang ringkas, jelas, dan actionable dalam format Markdown. Laporan harus mencakup:
    
    1.  **Ringkasan Eksekutif:** Mulai dengan kesimpulan utama dalam 2-3 kalimat. Jelaskan secara singkat tingkat risiko di wilayah ini dan apa pendorong utamanya.
    
    2.  **Analisis Faktor Lokal vs Global:** Bandingkan hasil LIME (spesifik untuk wilayah ini) dengan pola SHAP global. Jelaskan apakah wilayah ini mengikuti pola umum atau memiliki karakteristik unik.
    
    3.  **Analisis Faktor Peningkat Risiko:** Jelaskan MENGAPA faktor-faktor ini kemungkinan besar meningkatkan risiko. Hubungkan dengan teori ekonomi atau logika umum. Jika menemukan paradoks (misal: Upah Minimum tinggi meningkatkan risiko), berikan analisis mendalam tentang itu.
    
    4.  **Analisis Faktor Penurun Risiko:** Jelaskan MENGAPA faktor-faktor ini menjadi kekuatan bagi wilayah tersebut dan membantu menekan risiko.
    
    5.  **Rekomendasi Kebijakan:** Berdasarkan analisis lokal dan global, berikan 3-4 rekomendasi kebijakan yang konkret dan dapat ditindaklanjuti, dengan prioritas berdasarkan dampak yang diharapkan.

    Gunakan bahasa yang profesional namun mudah dipahami oleh audiens non-teknis. Jika ada data SHAP yang tersedia, pastikan untuk mengintegrasikan insight global dengan analisis lokal LIME.
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Anda adalah seorang analis ekonomi ahli."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Terjadi kesalahan saat menghasilkan analisis AI dari DeepSeek: {str(e)}"

# --- FUNGSI UTAMA STREAMLIT ---
def main():
    st.markdown('<div class="main-header">📊 Socioeconomic Risk Predictor with XAI</div>', unsafe_allow_html=True)

    # --- MODIFIKASI: Konfigurasi Klien API untuk DeepSeek ---
    try:
        client = openai.OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1"
        )
        llm_configured = True
    except Exception as e:
        st.warning("⚠️ LLM DeepSeek tidak terkonfigurasi. Fitur Analisis AI tidak akan tersedia. Pastikan Anda mengatur DEEPSEEK_API_KEY di st.secrets.", icon="🤖")
        client = None
        llm_configured = False

    model = load_model()
    if model is None:
        st.error("Gagal memuat model prediksi. Pastikan file pipeline ada.")
        return

    # Inisialisasi session state (tidak ada perubahan)
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
    if 'shap_analysis' not in st.session_state:
        st.session_state.shap_analysis = None

    with st.sidebar:
        st.markdown('<div class="sidebar-header">📁 Data Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload file CSV data sosioekonomi", type=['csv'])
        
        if uploaded_file:
            st.success("✅ File berhasil diunggah!")
        
        st.markdown("---")
        st.markdown("### 📋 Format Data yang Dibutuhkan")
        st.markdown("File CSV harus mengandung kolom:")
        
        with st.expander("📊 Daftar Kolom Wajib", expanded=False):
            for i, col in enumerate(MODEL_FEATURES, 1):
                st.write(f"{i}. `{col}`")
            st.write("---")
            st.write("**Kolom Metadata:**")
            st.write("- `kabupaten_kota`")
            st.write("- `tahun`") 
            st.write("- `kuartal`")
            st.write("- `Proksi Inflasi`")
        
        st.markdown("---")
        st.markdown("### 🤖 Fitur AI")
        st.info("💡 **Analisis Naratif AI:** Setelah prediksi, Anda dapat menggunakan AI untuk mendapatkan analisis mendalam yang menggabungkan insight LIME dan SHAP.")
        
        st.markdown("---")
        st.markdown("### ℹ️ Tentang Model")
        st.markdown("""
        Model ini menggunakan **XGBoost** dengan fitur explainable AI:
        - **SHAP**: Analisis global feature importance
        - **LIME**: Penjelasan prediksi individual  
        - **AI Analysis**: Interpretasi naratif yang mudah dipahami
        """)
    
    if uploaded_file:
        df = validate_csv(uploaded_file)
        if df is not None:
            st.success(f"Data berhasil divalidasi! Memuat {len(df)} baris.")
            X = df[MODEL_FEATURES].copy()
            # ... (logika validasi kolom tidak ada perubahan) ...
            
            if 'predictions_made' not in st.session_state: st.session_state.predictions_made = False
            
            if st.button("🚀 Buat Prediksi", type="primary"):
                with st.spinner("Membuat prediksi dan menghasilkan penjelasan..."):
                    try:
                        predictions = model.predict(X)
                        df_with_predictions = df.copy()
                        df_with_predictions['Risk_Score'] = predictions
                        st.session_state.predictions = predictions
                        st.session_state.df_with_predictions = df_with_predictions
                        st.session_state.predictions_made = True
                    except Exception as e:
                        st.error(f"Error saat prediksi: {e}")
                        st.session_state.predictions_made = False
            
            if st.session_state.predictions_made:
                df_with_predictions = st.session_state.df_with_predictions
                predictions = st.session_state.predictions

                st.markdown('<div class="section-header">📈 Hasil Prediksi</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Rata-rata Skor Risiko", f"{np.mean(predictions):.3f}")
                col2.metric("Skor Risiko Minimum", f"{np.min(predictions):.3f}")
                col3.metric("Skor Risiko Maksimum", f"{np.max(predictions):.3f}")

                with st.expander("📊 Tabel Prediksi Detail", expanded=False):
                    st.dataframe(df_with_predictions[['kabupaten_kota', 'tahun', 'kuartal', 'Risk_Score']].sort_values('Risk_Score', ascending=False))
                
                # --- Bagian SHAP ---
                st.markdown('<div class="section-header">🔍 Wawasan Explainable AI (XAI)</div>', unsafe_allow_html=True)
                st.markdown("#### 📊 Global Feature Importance (SHAP)")
                st.markdown("*Memahami dampak global setiap fitur terhadap prediksi model*")
                
                # Generate SHAP explanations
                with st.spinner("Menghasilkan analisis SHAP..."):
                    shap_values = generate_shap_explanations(model, X, MODEL_FEATURES)
                
                if shap_values is not None:
                    # Store SHAP values in session state for later use
                    st.session_state.shap_values = shap_values
                    st.session_state.X_for_shap = X
                    
                    # Global feature importance
                    try:
                        # Calculate mean absolute SHAP values for global importance
                        if hasattr(shap_values, 'values'):
                            shap_values_array = shap_values.values
                        else:
                            shap_values_array = shap_values
                            
                        mean_abs_shap = np.abs(shap_values_array).mean(0)
                        feature_importance_df = pd.DataFrame({
                            'Feature': MODEL_FEATURES,
                            'Importance': mean_abs_shap
                        }).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Bar plot of feature importance
                            fig, ax = plt.subplots(figsize=(10, 8))
                            bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
                            ax.set_yticks(range(len(feature_importance_df)))
                            ax.set_yticklabels(feature_importance_df['Feature'])
                            ax.set_xlabel('Mean |SHAP value|')
                            ax.set_title('Global Feature Importance (SHAP)')
                            ax.grid(axis='x', alpha=0.3)
                            
                            # Color bars based on importance
                            max_importance = feature_importance_df['Importance'].max()
                            for i, bar in enumerate(bars):
                                bar.set_color(plt.cm.viridis(feature_importance_df.iloc[i]['Importance'] / max_importance))
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.markdown("**🏆 Top 10 Fitur Paling Berpengaruh:**")
                            for idx, row in feature_importance_df.head(10).iterrows():
                                st.write(f"**{idx+1}.** {row['Feature']}")
                                st.write(f"   Skor: {row['Importance']:.4f}")
                                st.write("---")
                        
                        # Filter options for SHAP analysis
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            region_options = ["All"] + sorted(df_with_predictions['kabupaten_kota'].unique().tolist())
                            shap_region = st.selectbox("Filter by Region (SHAP)", options=region_options, 
                                                     index=region_options.index(st.session_state.shap_selected_region))
                            st.session_state.shap_selected_region = shap_region
                        
                        with col2:
                            year_options = ["All"] + sorted(df_with_predictions['tahun'].unique().tolist())
                            shap_year = st.selectbox("Filter by Year (SHAP)", options=year_options,
                                                   index=year_options.index(st.session_state.shap_selected_year))
                            st.session_state.shap_selected_year = shap_year
                        
                        with col3:
                            quarter_options = ["All"] + sorted(df_with_predictions['kuartal'].unique().tolist())
                            shap_quarter = st.selectbox("Filter by Quarter (SHAP)", options=quarter_options,
                                                      index=quarter_options.index(st.session_state.shap_selected_quarter))
                            st.session_state.shap_selected_quarter = shap_quarter
                        
                        # Filter data based on selection
                        shap_filtered_df = df_with_predictions.copy()
                        shap_filter_indices = list(range(len(df_with_predictions)))
                        
                        if shap_region != "All":
                            mask = shap_filtered_df['kabupaten_kota'] == shap_region
                            shap_filtered_df = shap_filtered_df[mask]
                            shap_filter_indices = [i for i, include in enumerate(mask) if include]
                        
                        if shap_year != "All":
                            mask = shap_filtered_df['tahun'] == shap_year
                            shap_filtered_df = shap_filtered_df[mask]
                            shap_filter_indices = [shap_filter_indices[i] for i, include in enumerate(mask) if include]
                        
                        if shap_quarter != "All":
                            mask = shap_filtered_df['kuartal'] == shap_quarter
                            shap_filtered_df = shap_filtered_df[mask]
                            shap_filter_indices = [shap_filter_indices[i] for i, include in enumerate(mask) if include]
                        
                        if len(shap_filter_indices) > 0:
                            # Summary plot for filtered data
                            if len(shap_filter_indices) > 1:
                                st.markdown("**📈 SHAP Summary Plot (Data Terfilter):**")
                                try:
                                    # Try to create SHAP summary plot
                                    shap_subset = shap_values[shap_filter_indices]
                                    shap.summary_plot(shap_subset, X.iloc[shap_filter_indices], 
                                                    feature_names=MODEL_FEATURES, show=False)
                                    st.pyplot(plt.gcf())
                                    plt.close()
                                except Exception as e:
                                    st.warning(f"SHAP summary plot tidak dapat dibuat: {e}")
                                    st.info("Menampilkan visualisasi alternatif...")
                                    
                                    # Alternative visualization: Feature importance bar plot
                                    if hasattr(shap_values, 'values'):
                                        filtered_shap_values = shap_values.values[shap_filter_indices]
                                    else:
                                        filtered_shap_values = shap_values[shap_filter_indices]
                                    
                                    mean_abs_shap_filtered = np.abs(filtered_shap_values).mean(0)
                                    
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    feature_importance_filtered = pd.DataFrame({
                                        'Feature': MODEL_FEATURES,
                                        'Importance': mean_abs_shap_filtered
                                    }).sort_values('Importance', ascending=True)
                                    
                                    bars = ax.barh(range(len(feature_importance_filtered)), 
                                                 feature_importance_filtered['Importance'])
                                    ax.set_yticks(range(len(feature_importance_filtered)))
                                    ax.set_yticklabels(feature_importance_filtered['Feature'])
                                    ax.set_xlabel('Mean |SHAP value| (Filtered Data)')
                                    ax.set_title('Feature Importance for Filtered Data')
                                    ax.grid(axis='x', alpha=0.3)
                                    
                                    # Color bars
                                    max_importance = feature_importance_filtered['Importance'].max()
                                    for i, bar in enumerate(bars):
                                        bar.set_color(plt.cm.viridis(feature_importance_filtered.iloc[i]['Importance'] / max_importance))
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                            
                            # Calculate average SHAP values for filtered data
                            if hasattr(shap_values, 'values'):
                                filtered_shap_values = shap_values.values[shap_filter_indices]
                            else:
                                filtered_shap_values = shap_values[shap_filter_indices]
                            
                            avg_shap = np.mean(filtered_shap_values, axis=0)
                            
                            # Store filtered SHAP analysis for AI
                            shap_analysis = {
                                'positive_factors': [(MODEL_FEATURES[i], avg_shap[i]) for i in range(len(MODEL_FEATURES)) if avg_shap[i] > 0],
                                'negative_factors': [(MODEL_FEATURES[i], avg_shap[i]) for i in range(len(MODEL_FEATURES)) if avg_shap[i] < 0]
                            }
                            shap_analysis['positive_factors'].sort(key=lambda x: x[1], reverse=True)
                            shap_analysis['negative_factors'].sort(key=lambda x: x[1])
                            
                            st.session_state.shap_analysis = shap_analysis
                            
                            # Display top factors from SHAP
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**🔴 Faktor Global Peningkat Risiko (SHAP):**")
                                for feature, weight in shap_analysis['positive_factors'][:5]:
                                    st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Avg SHAP: <strong>+{weight:.4f}</strong></div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("**🟢 Faktor Global Penurun Risiko (SHAP):**")
                                for feature, weight in shap_analysis['negative_factors'][:5]:
                                    st.markdown(f'<div class="risk-decreasing"><strong>{feature}</strong><br/>Avg SHAP: <strong>{weight:.4f}</strong></div>', unsafe_allow_html=True)
                        
                        else:
                            st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
                    
                    except Exception as e:
                        st.error(f"Error dalam analisis SHAP: {e}")
                
                else:
                    st.warning("SHAP analysis tidak dapat dibuat. Melanjutkan dengan analisis LIME saja.")

                # --- Bagian LIME dengan Integrasi LLM ---
                st.markdown("#### 🔎 Penjelasan Lokal (LIME)")
                st.markdown("*Memahami prediksi individual untuk wilayah dan periode waktu tertentu*")
                
                col1, col2, col3 = st.columns(3)
                unique_regions = sorted(df_with_predictions['kabupaten_kota'].unique())
                unique_years = sorted(df_with_predictions['tahun'].unique())
                unique_quarters = sorted(df_with_predictions['kuartal'].unique())
                
                # ... (logika pemilihan instance LIME Anda tetap di sini) ...
                with col1:
                    st.session_state.selected_region = st.selectbox("Pilih Kabupaten/Kota", options=unique_regions, index=0 if st.session_state.selected_region is None else unique_regions.index(st.session_state.selected_region))
                with col2:
                    st.session_state.selected_year = st.selectbox("Pilih Tahun", options=unique_years, index=len(unique_years)-1 if st.session_state.selected_year is None else unique_years.index(st.session_state.selected_year))
                with col3:
                    st.session_state.selected_quarter = st.selectbox("Pilih Kuartal", options=unique_quarters, index=0 if st.session_state.selected_quarter is None else unique_quarters.index(st.session_state.selected_quarter))

                filtered_df = df_with_predictions[
                    (df_with_predictions['kabupaten_kota'] == st.session_state.selected_region) & 
                    (df_with_predictions['tahun'] == st.session_state.selected_year) & 
                    (df_with_predictions['kuartal'] == st.session_state.selected_quarter)
                ]
                
                if not filtered_df.empty:
                    instance_idx = filtered_df.index[0]
                    predicted_risk = predictions[instance_idx]
                    lime_exp = generate_lime_explanation(model, X, instance_idx, MODEL_FEATURES)
                    
                    if lime_exp:
                        st.markdown(f"**📋 Analisis Dampak Fitur untuk {st.session_state.selected_region} - Q{st.session_state.selected_quarter} {st.session_state.selected_year}**")
                        st.info(f"Skor Risiko Prediksi: **{predicted_risk:.3f}**")
                        
                        exp_list = lime_exp.as_list()
                        positive_features = sorted([(f, w) for f, w in exp_list if w > 0], key=lambda item: item[1], reverse=True)
                        negative_features = sorted([(f, w) for f, w in exp_list if w < 0], key=lambda item: item[1])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🔴 Faktor Peningkat Risiko:**")
                            for feature, weight in positive_features:
                                st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Dampak: <strong>+{weight:.3f}</strong></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown("**🟢 Faktor Penurun Risiko:**")
                            for feature, weight in negative_features:
                                st.markdown(f'<div class="risk-decreasing"><strong>{feature}</strong><br/>Dampak: <strong>{weight:.3f}</strong></div>', unsafe_allow_html=True)

                        # --- MODIFIKASI: Tombol dan Tampilan Analisis AI ---
                        st.markdown("---")
                        if llm_configured:
                            btn_key = f"ai_btn_{instance_idx}"
                            if st.button("🤖 Buat Analisis Naratif dengan AI", key=btn_key):
                                with st.spinner("🧠 AI sedang menyusun laporan analisis mendalam..."):
                                    # Get SHAP analysis if available
                                    shap_analysis = st.session_state.get('shap_analysis', None)
                                    
                                    narrative = generate_narrative_explanation(
                                        client, # Menggunakan klien DeepSeek
                                        st.session_state.selected_region,
                                        st.session_state.selected_year,
                                        st.session_state.selected_quarter,
                                        predicted_risk,
                                        positive_features,
                                        negative_features,
                                        shap_analysis  # Tambahkan data SHAP
                                    )
                                    st.session_state[f'narrative_{instance_idx}'] = narrative
                            
                            # Tampilkan narasi jika ada
                            if f'narrative_{instance_idx}' in st.session_state:
                                st.markdown("### 💡 Laporan Analisis AI")
                                # Add information about data sources
                                st.info("📊 **Sumber Analisis:** Laporan ini mengombinasikan analisis LIME (spesifik wilayah) dan SHAP (pola global) untuk memberikan pemahaman yang komprehensif.")
                                st.markdown(st.session_state[f'narrative_{instance_idx}'], unsafe_allow_html=True)
                        # --- Akhir Modifikasi ---
                    
                    else:
                        st.warning("Penjelasan LIME tidak dapat dibuat. Menampilkan nilai fitur mentah.")
                        
                        # Fallback: Display raw feature values
                        st.markdown("**📊 Nilai Fitur untuk Instance Ini:**")
                        instance_data = X.iloc[instance_idx]
                        
                        col1, col2 = st.columns(2)
                        mid_point = len(MODEL_FEATURES) // 2
                        
                        with col1:
                            for feature in MODEL_FEATURES[:mid_point]:
                                st.metric(feature, f"{instance_data[feature]:.3f}")
                        
                        with col2:
                            for feature in MODEL_FEATURES[mid_point:]:
                                st.metric(feature, f"{instance_data[feature]:.3f}")
                        
                        # Still show AI analysis if available, but without LIME factors
                        if llm_configured:
                            st.markdown("---")
                            fallback_btn_key = f"ai_fallback_btn_{instance_idx}"
                            if st.button("🤖 Buat Analisis Dasar dengan AI", key=fallback_btn_key):
                                with st.spinner("🧠 AI sedang menganalisis berdasarkan data mentah..."):
                                    # Create simple positive/negative factors based on feature values
                                    median_values = X.median()
                                    simple_factors = []
                                    
                                    for feature in MODEL_FEATURES:
                                        value = instance_data[feature]
                                        median_val = median_values[feature]
                                        diff = (value - median_val) / median_val if median_val != 0 else 0
                                        simple_factors.append((feature, diff))
                                    
                                    # Split into positive and negative based on deviation from median
                                    simple_positive = [(f, w) for f, w in simple_factors if w > 0]
                                    simple_negative = [(f, w) for f, w in simple_factors if w < 0]
                                    
                                    shap_analysis = st.session_state.get('shap_analysis', None)
                                    
                                    narrative = generate_narrative_explanation(
                                        client,
                                        st.session_state.selected_region,
                                        st.session_state.selected_year,
                                        st.session_state.selected_quarter,
                                        predicted_risk,
                                        simple_positive[:5],  # Top 5 above median
                                        simple_negative[:5], # Top 5 below median
                                        shap_analysis
                                    )
                                    st.session_state[f'fallback_narrative_{instance_idx}'] = narrative
                            
                            if f'fallback_narrative_{instance_idx}' in st.session_state:
                                st.markdown("### 💡 Laporan Analisis AI (Berbasis Data Mentah)")
                                st.warning("⚠️ Analisis ini berbasis perbandingan dengan nilai median karena LIME tidak tersedia.")
                                st.markdown(st.session_state[f'fallback_narrative_{instance_idx}'], unsafe_allow_html=True)
                
                else:
                    st.warning(f"Tidak ada data untuk {st.session_state.selected_region} pada Q{st.session_state.selected_quarter} {st.session_state.selected_year}")

                # --- Bagian Visualisasi dan Ekspor ---
                st.markdown('<div class="section-header">📊 Visualisasi dan Ekspor Data</div>', unsafe_allow_html=True)
                
                # Distribution plot
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Distribusi Skor Risiko**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(np.mean(predictions), color='red', linestyle='--', label=f'Mean: {np.mean(predictions):.3f}')
                    ax.axvline(np.median(predictions), color='green', linestyle='--', label=f'Median: {np.median(predictions):.3f}')
                    ax.set_xlabel('Risk Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Risk Scores')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("**🗺️ Risiko per Wilayah**")
                    region_risk = df_with_predictions.groupby('kabupaten_kota')['Risk_Score'].mean().sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    y_pos = np.arange(len(region_risk))
                    bars = ax.barh(y_pos, region_risk.values)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(region_risk.index, fontsize=8)
                    ax.set_xlabel('Average Risk Score')
                    ax.set_title('Average Risk Score by Region')
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Color code bars
                    for i, bar in enumerate(bars):
                        bar.set_color(plt.cm.Reds(region_risk.values[i] / region_risk.max()))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Export functionality
                st.markdown("**💾 Ekspor Hasil Prediksi**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = df_with_predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"risk_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df_with_predictions.to_excel(writer, sheet_name='Predictions', index=False)
                        
                        # Add summary sheet
                        summary_df = pd.DataFrame({
                            'Metric': ['Mean Risk Score', 'Min Risk Score', 'Max Risk Score', 'Std Risk Score'],
                            'Value': [np.mean(predictions), np.min(predictions), np.max(predictions), np.std(predictions)]
                        })
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"risk_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    # Generate summary report
                    summary_report = f"""
                    # Risk Prediction Summary Report
                    
                    **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                    **Total Records:** {len(df_with_predictions)}
                    
                    ## Risk Score Statistics
                    - **Mean:** {np.mean(predictions):.4f}
                    - **Median:** {np.median(predictions):.4f}
                    - **Standard Deviation:** {np.std(predictions):.4f}
                    - **Min:** {np.min(predictions):.4f}
                    - **Max:** {np.max(predictions):.4f}
                    
                    ## Top 5 Highest Risk Regions
                    {region_risk.head().to_string()}
                    
                    ## Top 5 Lowest Risk Regions  
                    {region_risk.tail().to_string()}
                    """
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=summary_report,
                        file_name=f"risk_summary_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
    else:
        # --- Halaman Sambutan (tidak ada perubahan) ---
        st.markdown("""
        <div class="info-card">
            <h3>👋 Selamat Datang di Prediktor Risiko Sosioekonomi!</h3>
            <p><em>Analitik machine learning canggih untuk penilaian risiko sosioekonomi. Unggah data Anda di sidebar untuk memulai.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
