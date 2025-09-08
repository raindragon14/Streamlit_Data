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
    page_title="Small Business Failure Risk Predictor",
    page_icon="🏪",
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
    **Analisis Global (SHAP) - Pola Umum Kegagalan Usaha Kecil:**
    **Faktor Global Peningkat Risiko Kegagalan Usaha Kecil:**
    {shap_pos_str}
    **Faktor Global Penurun Risiko Kegagalan Usaha Kecil:**
    {shap_neg_str}
    """
    
    prompt_content = f"""
    Anda adalah seorang analis ekonomi dan konsultan Usaha Kecil ahli yang ditugaskan untuk menjelaskan prediksi risiko kegagalan usaha kecil untuk seorang kepala daerah, pembuat kebijakan, atau lembaga pemberdayaan Usaha Kecil.
    Catatan : "Persentase Penduduk" adalah (jumlah penduduk di wilayah itu)/jumlah penduduk Jawa Timur lalu dikalikan dengan 100%
    **Konteks Analisis:**
    - **Wilayah:** {region}
    - **Periode:** Kuartal {quarter}, Tahun {year}
    - **Skor Risiko Kegagalan Usaha Kecil:** {risk_score:.3f} (skor lebih tinggi berarti risiko kegagalan usaha kecil lebih besar)

    **Faktor-faktor Kunci dari Model (berdasarkan LIME - Analisis Lokal untuk Wilayah Ini):**
    **Faktor Peningkat Risiko Kegagalan Usaha Kecil:**
    {pos_factors_str}
    **Faktor Penurun Risiko Kegagalan Usaha Kecil:**
    {neg_factors_str}
    {shap_section}

    **Tugas Anda:**
    Tulis laporan analisis yang ringkas, jelas, dan actionable dalam format Markdown. Laporan harus mencakup:
    
    1.  **Ringkasan Eksekutif:** Mulai dengan kesimpulan utama dalam 2-3 kalimat. Jelaskan secara singkat tingkat risiko kegagalan Usaha Kecil di wilayah ini dan apa pendorong utamanya.
    
    2.  **Analisis Faktor Lokal vs Global:** Bandingkan hasil LIME (spesifik untuk wilayah ini) dengan pola SHAP global. Jelaskan apakah wilayah ini mengikuti pola umum kegagalan Usaha Kecil atau memiliki karakteristik unik.
    
    3.  **Analisis Faktor Peningkat Risiko:** Jelaskan MENGAPA faktor-faktor ini kemungkinan besar meningkatkan risiko kegagalan Usaha Kecil. Hubungkan dengan teori ekonomi Usaha Kecil, akses permodalan, daya beli masyarakat, atau kondisi pasar lokal.
    
    4.  **Analisis Faktor Penurun Risiko:** Jelaskan MENGAPA faktor-faktor ini menjadi kekuatan pendukung keberhasilan Usaha Kecil di wilayah tersebut dan membantu menekan risiko kegagalan.
    
    5.  **Rekomendasi Kebijakan Usaha Kecil:** Berdasarkan analisis lokal dan global, berikan 3-4 rekomendasi kebijakan yang konkret dan dapat ditindaklanjuti untuk pemberdayaan Usaha Kecil, dengan prioritas berdasarkan dampak yang diharapkan terhadap pengurangan risiko kegagalan usaha kecil.

    Gunakan bahasa yang profesional namun mudah dipahami oleh audiens non-teknis. Fokus pada konteks Usaha Kecil, seperti akses permodalan, daya beli masyarakat, infrastruktur bisnis, dan ekosistem kewirausahaan. Jika ada data SHAP yang tersedia, pastikan untuk mengintegrasikan insight global dengan analisis lokal LIME.
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Anda adalah seorang analis ekonomi dan konsultan Usaha Kecil ahli."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Terjadi kesalahan saat menghasilkan analisis AI dari DeepSeek: {str(e)}"

def generate_narrative_explanation_custom(client, region, year, quarter, risk_score, positive_features, negative_features, shap_analysis=None, format_instruction="", language="Bahasa Indonesia", temperature=0.7, max_tokens=1500):
    """Generate a custom narrative explanation with specific parameters."""
    if not client:
        return "Layanan AI tidak dikonfigurasi."

    pos_factors_str = "\n".join([f"- `{factor}` (Kontribusi: {weight:.3f})" for factor, weight in positive_features])
    neg_factors_str = "\n".join([f"- `{factor}` (Kontribusi: {weight:.3f})" for factor, weight in negative_features])
    
    # Add SHAP analysis if available
    shap_section = ""
    if shap_analysis:
        shap_pos_str = "\n".join([f"- `{factor}` (Rata-rata SHAP: {weight:.4f})" for factor, weight in shap_analysis['positive_factors'][:3]])
        shap_neg_str = "\n".join([f"- `{factor}` (Rata-rata SHAP: {weight:.4f})" for factor, weight in shap_analysis['negative_factors'][:3]])
        
        shap_section = f"""
    **Pola Global Kegagalan Usaha Kecil (SHAP):**
    **Peningkat Risiko Global:** {shap_pos_str}
    **Penurun Risiko Global:** {shap_neg_str}
    """
    
    # Language instruction
    language_instruction = ""
    if language == "English":
        language_instruction = "Please write the analysis in English."
    elif language == "Mixed (ID/EN)":
        language_instruction = "Use mixed Indonesian and English, with technical terms in English but explanations in Indonesian."
    else:
        language_instruction = "Tulis analisis dalam Bahasa Indonesia."
    
    prompt_content = f"""
    Anda adalah analis ekonomi dan konsultan Usaha Kecil ahli. {language_instruction}
    
    **Analisis Risiko Kegagalan Usaha Kecil untuk:** {region} - Q{quarter} {year}
    **Skor Risiko Kegagalan Usaha Kecil:** {risk_score:.3f}
    
    **Faktor Lokal (LIME):**
    **Peningkat Risiko Kegagalan Usaha Kecil:** {pos_factors_str}
    **Penurun Risiko Kegagalan Usaha Kecil:** {neg_factors_str}
    {shap_section}
    
    **Instruksi Format:** {format_instruction}
    
    Buat analisis yang fokus dan actionable untuk pemberdayaan Usaha Kecil berdasarkan format yang diminta. Fokus pada faktor-faktor yang mempengaruhi keberhasilan atau kegagalan usaha kecil seperti akses permodalan, daya beli masyarakat, infrastruktur bisnis, dan dukungan pemerintah.
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"Anda adalah analis ekonomi dan konsultan Usaha Kecil ahli. {language_instruction}"},
                {"role": "user", "content": prompt_content}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# --- FUNGSI UTAMA STREAMLIT ---
def main():
    st.markdown('<div class="main-header">🏪 Small Business Failure Risk Predictor with XAI</div>', unsafe_allow_html=True)

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
        uploaded_file = st.file_uploader("Upload file CSV data sosioekonomi untuk analisis risiko usaha kecil", type=['csv'])
        
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
        st.markdown("### 🤖 Fitur DeepSeek AI")
        st.info("💡 **Analisis AI untuk Risiko Usaha Kecil:** Aplikasi ini menggunakan DeepSeek untuk analisis mendalam risiko kegagalan usaha kecil dengan fitur:")
        st.markdown("""
        - 🎯 **Analisis Individual**: Per wilayah dengan LIME + SHAP untuk prediksi kegagalan usaha kecil
        - 📊 **Analisis Massal**: Batch analysis untuk multiple regions
        - ⚙️ **Parameter Custom**: Temperature, max tokens, bahasa
        - 📄 **Multiple Format**: Ringkasan, detail, atau fokus kebijakan Usaha Kecil
        - 🌍 **Multi-bahasa**: Indonesia, English, atau Mixed
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ Tentang Model")
        st.markdown("""
        **🔬 XAI Pipeline untuk Prediksi Kegagalan Usaha Kecil:**
        - **XGBoost**: Model prediksi utama untuk risiko kegagalan Usaha Kecil
        - **SHAP**: Global feature importance & patterns yang mempengaruhi usaha kecil
        - **LIME**: Local explanations per wilayah untuk kondisi spesifik Usaha Kecil
        - **DeepSeek**: AI narrative analysis & insights untuk strategi pemberdayaan Usaha Kecil
        
        **📈 Workflow:**
        1. Upload data → Prediksi risiko kegagalan usaha kecil
        2. SHAP analysis → Pola global faktor kegagalan Usaha Kecil
        3. LIME analysis → Penjelasan lokal per wilayah
        4. DeepSeek AI → Insights dan rekomendasi pemberdayaan Usaha Kecil
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

                st.markdown('<div class="section-header">📈 Hasil Prediksi Risiko Kegagalan Usaha Kecil</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Rata-rata Skor Risiko Kegagalan Usaha Kecil", f"{np.mean(predictions):.3f}")
                col2.metric("Skor Risiko Minimum", f"{np.min(predictions):.3f}")
                col3.metric("Skor Risiko Maksimum", f"{np.max(predictions):.3f}")

                with st.expander("📊 Tabel Prediksi Detail", expanded=False):
                    st.dataframe(df_with_predictions[['kabupaten_kota', 'tahun', 'kuartal', 'Risk_Score']].sort_values('Risk_Score', ascending=False).rename(columns={'Risk_Score': 'Risiko_Kegagalan_Usaha_Kecil'}))
                
                # --- Bagian SHAP ---
                st.markdown('<div class="section-header">🔍 Wawasan Explainable AI (XAI) untuk Risiko Usaha Kecil</div>', unsafe_allow_html=True)
                st.markdown("#### 📊 Global Feature Importance (SHAP)")
                st.markdown("*Memahami dampak global setiap faktor sosioekonomi terhadap risiko kegagalan usaha kecil*")
                
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
                            ax.set_title('Global Feature Importance untuk Risiko Kegagalan Usaha Kecil (SHAP)')
                            ax.grid(axis='x', alpha=0.3)
                            
                            # Color bars based on importance
                            max_importance = feature_importance_df['Importance'].max()
                            for i, bar in enumerate(bars):
                                bar.set_color(plt.cm.viridis(feature_importance_df.iloc[i]['Importance'] / max_importance))
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.markdown("**🏆 Top 10 Faktor Paling Berpengaruh terhadap Kegagalan Usaha Kecil:**")
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
                                st.markdown("**🔴 Faktor Global Peningkat Risiko Kegagalan Usaha Kecil (SHAP):**")
                                for feature, weight in shap_analysis['positive_factors'][:5]:
                                    st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Avg SHAP: <strong>+{weight:.4f}</strong></div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("**🟢 Faktor Global Penurun Risiko Kegagalan Usaha Kecil (SHAP):**")
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
                st.markdown("*Memahami prediksi individual risiko kegagalan Usaha Kecil untuk wilayah dan periode waktu tertentu*")
                
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
                        st.markdown(f"**📋 Analisis Dampak Faktor Risiko Usaha Kecil untuk {st.session_state.selected_region} - Q{st.session_state.selected_quarter} {st.session_state.selected_year}**")
                        st.info(f"Skor Risiko Kegagalan Usaha Kecil: **{predicted_risk:.3f}**")
                        
                        exp_list = lime_exp.as_list()
                        positive_features = sorted([(f, w) for f, w in exp_list if w > 0], key=lambda item: item[1], reverse=True)
                        negative_features = sorted([(f, w) for f, w in exp_list if w < 0], key=lambda item: item[1])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🔴 Faktor Peningkat Risiko Kegagalan Usaha Kecil:**")
                            for feature, weight in positive_features:
                                st.markdown(f'<div class="risk-increasing"><strong>{feature}</strong><br/>Dampak: <strong>+{weight:.3f}</strong></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown("**🟢 Faktor Penurun Risiko Kegagalan Usaha Kecil:**")
                            for feature, weight in negative_features:
                                st.markdown(f'<div class="risk-decreasing"><strong>{feature}</strong><br/>Dampak: <strong>{weight:.3f}</strong></div>', unsafe_allow_html=True)

                        # --- MODIFIKASI: Tombol dan Tampilan Analisis AI ---
                        st.markdown("---")
                        if llm_configured:
                            btn_key = f"ai_btn_{instance_idx}"
                            if st.button("🤖 Buat Analisis Naratif Risiko Usaha Kecil dengan AI", key=btn_key):
                                with st.spinner("🧠 AI sedang menyusun laporan analisis risiko kegagalan Usaha Kecil..."):
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
                                st.markdown("### 💡 Laporan Analisis AI - Risiko Kegagalan Usaha Kecil")
                                # Add information about data sources
                                st.info("📊 **Sumber Analisis:** Laporan ini mengombinasikan analisis LIME (spesifik wilayah) dan SHAP (pola global) untuk memberikan pemahaman komprehensif tentang risiko kegagalan usaha kecil.")
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
                            if st.button("🤖 Buat Analisis Dasar Risiko Usaha Kecil dengan AI", key=fallback_btn_key):
                                with st.spinner("🧠 AI sedang menganalisis risiko Usaha Kecil berdasarkan data mentah..."):
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
                                st.markdown("### 💡 Laporan Analisis AI - Risiko Kegagalan Usaha Kecil (Berbasis Data Mentah)")
                                st.warning("⚠️ Analisis ini berbasis perbandingan dengan nilai median karena LIME tidak tersedia.")
                                st.markdown(st.session_state[f'fallback_narrative_{instance_idx}'], unsafe_allow_html=True)
                
                else:
                    st.warning(f"Tidak ada data untuk {st.session_state.selected_region} pada Q{st.session_state.selected_quarter} {st.session_state.selected_year}")

                # --- Bagian Konfigurasi dan Penggunaan LLM DeepSeek ---
                st.markdown('<div class="section-header">🤖 Konfigurasi LLM DeepSeek</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**🔧 Status Konfigurasi DeepSeek**")
                    if llm_configured:
                        st.success("✅ DeepSeek API terkonfigurasi dengan baik")
                        st.info("🎯 **Model:** deepseek-chat")
                        st.info("🌐 **Endpoint:** https://api.deepseek.com/v1")
                        
                        # Test connection button
                        if st.button("🔍 Test Koneksi DeepSeek"):
                            with st.spinner("Testing koneksi ke DeepSeek API..."):
                                try:
                                    test_response = client.chat.completions.create(
                                        model="deepseek-chat",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": "Hello, can you confirm this connection is working?"}
                                        ],
                                        max_tokens=50,
                                        temperature=0.3
                                    )
                                    st.success("🎉 Koneksi berhasil! DeepSeek merespons dengan baik.")
                                    st.code(test_response.choices[0].message.content)
                                except Exception as e:
                                    st.error(f"❌ Test koneksi gagal: {e}")
                    else:
                        st.error("❌ DeepSeek API belum dikonfigurasi")
                        st.markdown("""
                        **Cara Konfigurasi:**
                        1. Dapatkan API key dari [DeepSeek Platform](https://platform.deepseek.com/)
                        2. Tambahkan ke `st.secrets`:
                        ```toml
                        [secrets]
                        DEEPSEEK_API_KEY = "your-api-key-here"
                        ```
                        """)
                
                with col2:
                    st.markdown("**⚙️ Parameter LLM**")
                    
                    # LLM Parameters
                    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.1, 
                                          help="Kontrol kreativitas respons (0=deterministik, 1=kreatif)")
                    max_tokens = st.slider("📏 Max Tokens", 500, 3000, 1500, 100,
                                         help="Maksimum panjang respons AI")
                    
                    # Store in session state
                    st.session_state.llm_temperature = temperature
                    st.session_state.llm_max_tokens = max_tokens
                    
                    # Analysis language preference
                    analysis_language = st.selectbox("🌍 Bahasa Analisis", 
                                                    ["Bahasa Indonesia", "English", "Mixed (ID/EN)"],
                                                    help="Bahasa untuk laporan analisis AI")
                    st.session_state.analysis_language = analysis_language
                
                # Bulk Analysis Feature
                st.markdown("**📊 Analisis Massal Risiko Usaha Kecil dengan DeepSeek**")
                st.markdown("*Buat analisis AI untuk risiko kegagalan Usaha Kecil di semua wilayah atau kategori tertentu*")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    bulk_analysis_type = st.selectbox("Jenis Analisis Massal",
                                                     ["Top 5 Risiko Kegagalan Usaha Kecil Tertinggi", "Top 5 Risiko Kegagalan Usaha Kecil Terendah", 
                                                      "Semua Wilayah", "Per Tahun", "Per Kuartal"])
                
                with col2:
                    if bulk_analysis_type in ["Per Tahun", "Per Kuartal"]:
                        available_years = sorted(df_with_predictions['tahun'].unique())
                        selected_year_bulk = st.selectbox("Pilih Tahun", available_years)
                    else:
                        selected_year_bulk = None
                
                with col3:
                    analysis_format = st.selectbox("Format Output",
                                                  ["Ringkasan Eksekutif", "Laporan Detail", "Rekomendasi Kebijakan Usaha Kecil"])
                
                if st.button("🚀 Mulai Analisis Massal", type="primary"):
                    if not llm_configured:
                        st.error("DeepSeek API harus dikonfigurasi terlebih dahulu!")
                    else:
                        with st.spinner("🧠 DeepSeek sedang menganalisis risiko kegagalan Usaha Kecil secara massal..."):
                            bulk_results = []
                            
                            # Filter data based on analysis type
                            if bulk_analysis_type == "Top 5 Risiko Kegagalan Usaha Kecil Tertinggi":
                                top_risk = df_with_predictions.nlargest(5, 'Risk_Score')
                                analysis_data = top_risk
                            elif bulk_analysis_type == "Top 5 Risiko Kegagalan Usaha Kecil Terendah":
                                low_risk = df_with_predictions.nsmallest(5, 'Risk_Score')
                                analysis_data = low_risk
                            elif bulk_analysis_type == "Per Tahun":
                                analysis_data = df_with_predictions[df_with_predictions['tahun'] == selected_year_bulk]
                            elif bulk_analysis_type == "Per Kuartal":
                                analysis_data = df_with_predictions[df_with_predictions['tahun'] == selected_year_bulk]
                            else:  # Semua Wilayah
                                analysis_data = df_with_predictions.sample(min(10, len(df_with_predictions)))  # Limit to 10 for performance
                            
                            progress_bar = st.progress(0)
                            total_analyses = len(analysis_data)
                            
                            for idx, (_, row) in enumerate(analysis_data.iterrows()):
                                try:
                                    # Get LIME explanation for this instance
                                    instance_idx = row.name
                                    lime_exp = generate_lime_explanation(model, X, instance_idx, MODEL_FEATURES)
                                    
                                    if lime_exp:
                                        exp_list = lime_exp.as_list()
                                        positive_features = sorted([(f, w) for f, w in exp_list if w > 0], 
                                                                 key=lambda item: item[1], reverse=True)[:3]
                                        negative_features = sorted([(f, w) for f, w in exp_list if w < 0], 
                                                                 key=lambda item: item[1])[:3]
                                    else:
                                        # Fallback to simple analysis
                                        median_values = X.median()
                                        instance_data = X.iloc[instance_idx]
                                        simple_factors = []
                                        
                                        for feature in MODEL_FEATURES:
                                            value = instance_data[feature]
                                            median_val = median_values[feature]
                                            diff = (value - median_val) / median_val if median_val != 0 else 0
                                            simple_factors.append((feature, diff))
                                        
                                        positive_features = [(f, w) for f, w in simple_factors if w > 0][:3]
                                        negative_features = [(f, w) for f, w in simple_factors if w < 0][:3]
                                    
                                    # Generate AI analysis
                                    shap_analysis = st.session_state.get('shap_analysis', None)
                                    
                                    # Custom prompt based on format
                                    if analysis_format == "Ringkasan Eksekutif":
                                        format_instruction = "Buat ringkasan eksekutif singkat (maksimal 150 kata) yang fokus pada kesimpulan utama dan rekomendasi prioritas untuk pemberdayaan Usaha Kecil."
                                    elif analysis_format == "Rekomendasi Kebijakan Usaha Kecil":
                                        format_instruction = "Fokus pada rekomendasi kebijakan yang konkret dan actionable untuk pemberdayaan Usaha Kecil. Berikan 3-4 rekomendasi spesifik dengan prioritas implementasi."
                                    else:  # Laporan Detail
                                        format_instruction = "Buat laporan detail lengkap dengan analisis mendalam semua aspek yang mempengaruhi risiko kegagalan Usaha Kecil."
                                    
                                    custom_narrative = generate_narrative_explanation_custom(
                                        client,
                                        row['kabupaten_kota'],
                                        row['tahun'],
                                        row['kuartal'],
                                        row['Risk_Score'],
                                        positive_features,
                                        negative_features,
                                        shap_analysis,
                                        format_instruction,
                                        st.session_state.get('analysis_language', 'Bahasa Indonesia'),
                                        st.session_state.get('llm_temperature', 0.7),
                                        st.session_state.get('llm_max_tokens', 1500)
                                    )
                                    
                                    bulk_results.append({
                                        'region': row['kabupaten_kota'],
                                        'year': row['tahun'],
                                        'quarter': row['kuartal'],
                                        'risk_score': row['Risk_Score'],
                                        'analysis': custom_narrative
                                    })
                                    
                                except Exception as e:
                                    st.warning(f"Gagal menganalisis {row['kabupaten_kota']}: {e}")
                                
                                progress_bar.progress((idx + 1) / total_analyses)
                            
                            # Display results
                            st.success(f"✅ Analisis massal selesai! {len(bulk_results)} analisis berhasil dibuat.")
                            
                            # Save to session state
                            st.session_state.bulk_analysis_results = bulk_results
                            
                            # Display results with tabs
                            if bulk_results:
                                for i, result in enumerate(bulk_results):
                                    with st.expander(f"📋 {result['region']} - Q{result['quarter']} {result['year']} (Risiko Kegagalan Usaha Kecil: {result['risk_score']:.3f})"):
                                        st.markdown(result['analysis'])
                
                # Download bulk results
                if 'bulk_analysis_results' in st.session_state and st.session_state.bulk_analysis_results:
                    st.markdown("**💾 Download Hasil Analisis Massal Risiko Usaha Kecil**")
                    
                    # Prepare download data
                    bulk_text = f"# Laporan Analisis Risiko Kegagalan Usaha Kecil Massal - {bulk_analysis_type}\n"
                    bulk_text += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    
                    for result in st.session_state.bulk_analysis_results:
                        bulk_text += f"## {result['region']} - Q{result['quarter']} {result['year']}\n"
                        bulk_text += f"**Skor Risiko Kegagalan Usaha Kecil:** {result['risk_score']:.3f}\n\n"
                        bulk_text += result['analysis']
                        bulk_text += "\n\n---\n\n"
                    
                    st.download_button(
                        label="📄 Download Laporan Massal (Markdown)",
                        data=bulk_text,
                        file_name=f"bulk_analysis_usaha_kecil_{bulk_analysis_type.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
    else:
        # --- Halaman Sambutan (tidak ada perubahan) ---
        st.markdown("""
        <div class="info-card">
            <h3>👋 Selamat Datang di Prediktor Risiko Kegagalan Usaha Kecil!</h3>
            <p><em>Analitik machine learning canggih untuk penilaian risiko kegagalan Usaha Kecil berbasis faktor sosioekonomi. Unggah data Anda di sidebar untuk memulai analisis.</em></p>
            <p><strong>🎯 Fokus:</strong> Prediksi dan analisis risiko kegagalan usaha kecil untuk mendukung kebijakan pemberdayaan Usaha Kecil yang tepat sasaran.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
