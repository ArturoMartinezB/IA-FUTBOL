import streamlit as st
import sys
import os

# Agregar el directorio padre al path para importar desde otras carpetas
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Importar componentes locales
from components.ui_components import setup_page_config, render_header, render_sidebar
from components.video_handler import VideoUploader, VideoProcessor
from components.results_display import ResultsDisplay

# Importar desde tus carpetas existentes del proyecto
# from inference.your_detector import YourDetector
# from utils.your_processor import YourProcessor

def main():
    # Configurar página
    setup_page_config()
    
    # Renderizar header
    render_header()
    
    # Sidebar con configuraciones
    config = render_sidebar()
    
    # Manejador de video
    video_uploader = VideoUploader()
    video_processor = VideoProcessor() # En esta clase es donde se incluye el sistema de procesado
    results_display = ResultsDisplay()
    
    # Área principal - Upload de video
    uploaded_file = video_uploader.render_upload_section()
    
    if uploaded_file is not None:
        # Mostrar video original
        video_uploader.show_original_video(uploaded_file)
        
        # Procesar video
        if st.button("🚀 Procesar Video", type="primary"):
            # Guardar archivo temporal
            temp_path = video_uploader.save_temp_file(uploaded_file)
            
            with st.spinner("Procesando video..."):
                # Aquí llamarías a tu sistema de detección
                video_anotado, mapa_superior, estadisticas_deteccion, estadisticas_t1, estadisticas_t2 = video_processor.process_video(
                    temp_path, config
                )
                
                # Guardar en session state
                st.session_state.update({
                    'processed': True,
                    'video_anotado': video_anotado,
                    'mapa_superior': mapa_superior,
                    'estadisticas_deteccion': estadisticas_deteccion, 
                    'estadisticas_t1': estadisticas_t1,
                    'estadisticas_t2': estadisticas_t2
                })
    
    # Mostrar resultados si están disponibles
    if st.session_state.get('processed', False):
        results_display.render_results(config)

if __name__ == "__main__":
    main()