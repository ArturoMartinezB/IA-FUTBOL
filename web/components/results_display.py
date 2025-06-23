import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import sys
# Obtener el directorio raíz del proyecto de manera más robusta
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent  # Ajusta según tu estructura

# Agregar al path
sys.path.append(str(project_root))

class ResultsDisplay:
    """Clase para mostrar los resultados del análisis"""
    
    def render_results(self):
        """Renderizar todos los resultados"""
        st.markdown("---")
        st.markdown('<p class="section-header">🎯 Resultados del Análisis</p>', unsafe_allow_html=True)
        
        # Video anotado ocupando todo el ancho
        self._render_annotated_video()
        
        # Separador
        st.markdown("---")
        
        # Crear dos columnas para mapa superior y estadísticas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_top_view_video()
        
        with col2:
            self._render_statistics_t1()

        with col3:
            self._render_statistics_t2()
    
    def _render_annotated_video(self):
        """Renderizar sección del video anotado"""
        st.subheader("🎥 Video Anotado")
        
        # Placeholder para el video anotado
        st.info("Video con detecciones de jugadores")
        
        if os.path.exists(st.session_state['video_anotado']):
            video_path = st.session_state['video_anotado']
            
            # Crear versión compatible con Streamlit
            compatible_video_path = self._create_streamlit_compatible_video(video_path)
            
            if compatible_video_path:
                # Mostrar video compatible
                st.video(compatible_video_path)
            else:
                st.warning("No se pudo mostrar el video, pero puedes descargarlo")
            
            # Botón de descarga (del video original)
            with open(video_path, "rb") as file:
                video_bytes = file.read()
            
            st.download_button(
                label="📥 Descargar Video Anotado",
                data=video_bytes,
                file_name="video_anotado.mp4",
                mime="video/mp4",
                key="download_annotated"
            )
        else:
            st.error("Video anotado no encontrado")

    def _create_streamlit_compatible_video(self, original_path):
        """Crear una versión del video compatible con Streamlit"""
        import cv2
        import tempfile
        
        # Crear archivo temporal para la versión compatible
        with tempfile.NamedTemporaryFile(delete=False, suffix='_streamlit.mp4') as tmp_file:
            compatible_path = tmp_file.name
        
        # Verificar si ya existe la versión compatible
        if hasattr(st.session_state, 'compatible_video') and os.path.exists(st.session_state.compatible_video):
            return st.session_state.compatible_video
        
        try:
            # Abrir video original
            cap = cv2.VideoCapture(str(original_path))
            
            if not cap.isOpened():
                return None
            
            # Obtener propiedades
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Crear VideoWriter con codec compatible con Streamlit
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
            out = cv2.VideoWriter(compatible_path, fourcc, fps, (width, height))
            
            # Procesar frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frame_count += 1
                
                # Mostrar progreso cada 100 frames
                if frame_count % 100 == 0:
                    st.text(f"Recodificando para visualización: {frame_count} frames")
            
            # Limpiar
            cap.release()
            out.release()
            
            # Verificar que se creó correctamente
            if os.path.exists(compatible_path) and os.path.getsize(compatible_path) > 0:
                st.session_state.compatible_video = compatible_path
                st.success("✅ Video recodificado para visualización")
                return compatible_path
            else:
                return None
                
        except Exception as e:
            st.error(f"Error al recodificar video: {e}")
            return None


    def _render_top_view_video(self):
        """Renderizar sección del mapa superior"""
        st.subheader("🗺️ Mapa Superior")
        
        # Placeholder para el mapa superior
        st.info("Vista superior del campo con trayectorias")
        
        # Cuando tengas el video real, descomenta:
        if os.path.exists(st.session_state['mapa_superior']):
    # Mostrar video
            st.video(st.session_state['mapa_superior'])
            
            # Leer archivo una sola vez
            with open(st.session_state['mapa_superior'], "rb") as file:
                mapa_bytes = file.read()
            
            # Botón de descarga
            st.download_button(
                label="📥 Descargar Mapa Superior",
                data=mapa_bytes,
                file_name="mapa_superior.mp4",
                mime="video/mp4",
                key="download_map"
            )
        else:
            st.error("Mapa superior no encontrado")
        
       
            

    def _render_statistics_t1(self):
        """Renderizar sección de estadísticas"""
        st.subheader("📊 Estadísticas")
        
        stats = st.session_state['estadisticas_t1']
        
        # Mostrar métricas principales
        self._render_main_metrics(stats)
        
        # Mostrar datos completos
        st.subheader("Datos Completos")
        self._render_detailed_stats(stats)
        
        # Botón de descarga
        self._render_download_button(stats, 1)
    

    def _render_statistics_t2(self):
        """Renderizar sección de estadísticas"""
        st.subheader("📊 Estadísticas")
        
        stats = st.session_state['estadisticas_t2']
        
        # Mostrar métricas principales
        self._render_main_metrics(stats)
        
        # Mostrar datos completos
        st.subheader("Datos Completos")
        self._render_detailed_stats(stats)
        
        # Botón de descarga
        self._render_download_button(stats, 2)
    

    def _render_main_metrics(self, stats):
        """Renderizar métricas principales"""
        if isinstance(stats, dict):
            # Métricas destacadas
            key_metrics = {
                'jugadores_detectados': 'Jugadores',
                'precision_promedio': 'Precisión',
                'frames_totales': 'Frames',
                'tiempo_procesamiento': 'Tiempo (s)'
            }
            
            for key, label in key_metrics.items():
                if key in stats:
                    value = stats[key]
                    if key == 'precision_promedio':
                        st.metric(label, f"{value:.1%}")
                    elif key == 'tiempo_procesamiento':
                        st.metric(label, f"{value:.1f}s")
                    else:
                        st.metric(label, value)
    
    def _render_detailed_stats(self, stats):
        """Renderizar estadísticas detalladas"""
        if isinstance(stats, dict):
            # Mostrar como JSON expandible
            st.json(stats)
            
            # También como tabla si es útil
            if len(stats) > 0:
                df_stats = pd.DataFrame([
                    {'Métrica': k.replace('_', ' ').title(), 'Valor': v} 
                    for k, v in stats.items()
                ])
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
                
        elif isinstance(stats, list):
            # Si es una lista, mostrar como tabla
            df = pd.DataFrame(stats)
            st.dataframe(df, use_container_width=True)
        
        else:
            st.text(f"Tipo de estadísticas: {type(stats)}")
            st.write(stats)
    
    def _render_download_button(self, stats, team):

        """Renderizar botón de descarga de estadísticas"""
        key = "download_stats" + str(team)
        if st.button("📥 Descargar Estadísticas", key):
            # Crear datos para descarga
            if isinstance(stats, dict):
                json_data = json.dumps(stats, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Descargar JSON",
                    data=json_data,
                    file_name="estadisticas_deteccion.json",
                    mime="application/json",
                    key="download_json"
                )
            else:
                st.success("Preparando descarga...")
    
    def render_error_message(self, error_msg):
        """Renderizar mensaje de error"""
        st.error(f"❌ Error en el procesamiento: {error_msg}")
        
        with st.expander("Información de debug"):
            st.write("Verifica que:")
            st.write("- El video tenga un formato válido")
            st.write("- Tu sistema de detección esté funcionando")
            st.write("- Las rutas de salida sean correctas")