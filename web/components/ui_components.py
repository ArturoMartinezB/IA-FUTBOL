import streamlit as st

def setup_page_config():
    """Configurar la página de Streamlit"""
    st.set_page_config(
        page_title="Detección de Jugadores IA",
        page_icon="⚽",
        layout="wide"
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #2e7d32;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Renderizar el header principal"""
    st.markdown(
        '<h1 class="main-header">🤖 Sistema de Detección de Jugadores con IA</h1>', 
        unsafe_allow_html=True
    )

def render_sidebar():
    """Renderizar sidebar con configuraciones"""
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Parámetros de detección
        config = {
            'confidence_threshold': st.slider("Umbral de confianza", 0.0, 1.0, 0.5, 0.05),
            'show_trajectories': st.checkbox("Mostrar trayectorias", value=True),
            'show_ids': st.checkbox("Mostrar IDs", value=True),
            'model_type': st.selectbox("Modelo", ["YOLOv8", "YOLOv11", "Custom"])
        }
        
        st.markdown("---")
        st.subheader("📊 Info del Sistema")
        st.info("Modelo: Tu modelo personalizado\nEstado: Listo para procesar")
        
        return config

def render_section_header(title):
    """Renderizar header de sección"""
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)