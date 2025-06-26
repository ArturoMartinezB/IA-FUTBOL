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
        st.header("☑️ Seleccione")
        
        # Parámetros de detección
        config = {
        'resultados': {
            "Video anotado": st.sidebar.checkbox("🎥 Video anotado", value=True),
            "Mapeado del video": st.sidebar.checkbox("🗺️ Mapeado del video", value=True),
            "Estadísticas equipo 1": st.sidebar.checkbox("🧮 Estadísticas equipo 1", value=True),
            "Estadísticas equipo 2": st.sidebar.checkbox("🧮 Estadísticas equipo 2", value=False),
            "Estadísticas del procesamiento": st.sidebar.checkbox("🧠 Estadísticas del procesamiento", value=True),
        }
}

        
        st.markdown("---")
        st.subheader("📊 Info del Sistema")
        st.markdown("""
                    > **Modelo de detección por imagen:**  
                    > YOLOv9 Finetunned  
                    > **Modelo de detección de keypoints:**  
                    > YOLOv8s-pose Finetunned
                    """)

        
        return config

def render_section_header(title):
    """Renderizar header de sección"""
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)