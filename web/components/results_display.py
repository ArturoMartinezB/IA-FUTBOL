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
    
    def render_results(self, config):
        """Renderizar todos los resultados"""
        st.markdown("---")
        st.markdown('<p class="section-header">🎯 Resultados del Análisis</p>', unsafe_allow_html=True)
        
        resultados = config['resultados']

        # Video anotado ocupa todo el ancho si está activado
        if resultados.get('Video anotado', False):
            self._render_annotated_video()
            st.markdown("---")

        # Primer bloque: mapa y stats del procesamiento
        bloques = []
        if resultados.get('Mapeado del video', False):
            bloques.append(self._render_top_view_video)
        if resultados.get('Estadísticas del procesamiento', False):
            bloques.append(self._render_stats_procesamiento)

        if bloques:
            cols = st.columns(len(bloques))
            for col, render_fn in zip(cols, bloques):
                with col:
                    render_fn()

            st.markdown("---")

        # Segundo bloque: stats por equipo
        bloques_equipo = []
        if resultados.get('Estadísticas equipo 1', False):
            bloques_equipo.append(lambda: self._render_stats_equipo(1))
        if resultados.get('Estadísticas equipo 2', False):
            bloques_equipo.append(lambda: self._render_stats_equipo(2))

        if bloques_equipo:
            cols = st.columns(len(bloques_equipo))
            for col, render_fn in zip(cols, bloques_equipo):
                with col:
                    render_fn()

    
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
        '''if hasattr(st.session_state, 'compatible_video') and os.path.exists(st.session_state.compatible_video):
            return st.session_state.compatible_video'''
        
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

            st.text(f"🔄 Recodificando para visualización")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frame_count += 1
                
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
            st.video(self._create_streamlit_compatible_video(st.session_state['mapa_superior']))
            
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
        
       
    def _render_stats_procesamiento(self):
        st.subheader("⚙️ Estadísticas del procesamiento")

        etiquetas = {
            'possession_1': 'Posesión equipo 1',
            'possession_2': 'Posesión equipo 2',
            'passes': 'Pases detectados',
            'detections': 'Detecciones totales',
            'players_detections': 'Detecciones de jugadores',
            'ball_detections': 'Detecciones de balón',
            'ball_interpolations': 'Interpolaciones de balón',
            'detection_time': 'Tiempo total de detección (s)',
            'time_per_frame_detected': 'Tiempo por frame (detección)',
            'keypoint_time': 'Tiempo total keypoints (s)',
            'time_per_frame_keypoint': 'Tiempo por frame (keypoints)'
        }
        
        total_stats = st.session_state['estadisticas_proceso']
        for clave, valor in total_stats.items():
            etiqueta = etiquetas.get(clave, clave)
            st.metric(label=etiqueta, value=round(valor, 2) if isinstance(valor, float) else valor)

        st.download_button(
            label="📥 Descargar estadísticas del procesamiento",
            data=str(total_stats),
            file_name="estadisticas_procesamiento.txt"
        )

    def _render_stats_equipo(self, equipo=1):
        # Input para editar el nombre del equipo
        nombre_equipo = st.text_input(
            "Nombre del equipo:", 
            value=f"Equipo {equipo}", 
            key=f"nombre_equipo_{equipo}"
        )
        
        # Obtener lista de stats del equipo desde el session_state
        stats_dicc = st.session_state.get(f'estadisticas_t{equipo}', {})
        stats_jugadores = stats_dicc['stats_sheets']

        # Convertir color RGB a CSS
        team_color = stats_dicc['color']  # Por ejemplo (255, 0, 0)
        rgb_css_color = f"rgb({team_color[2]}, {team_color[1]}, {team_color[0]})"

        # Subheader con cuadrado de color
        color_box = f'<span style="display:inline-block; width:16px; height:16px; background-color:{rgb_css_color}; border:1px solid #000; margin-right:10px;"></span>'
        st.markdown(f"## {color_box}📊 Estadísticas {nombre_equipo}", unsafe_allow_html=True)


        if not stats_jugadores:
            st.warning("No hay estadísticas de jugadores disponibles.")
            return

        # Crear DataFrame y preparar para edición
        df_stats = pd.DataFrame(stats_jugadores)
        df_stats = df_stats.sort_values(by="dorsal")
        
        # Añadir columna de nombre de jugador si no existe
        if 'nombre_jugador' not in df_stats.columns:
            df_stats.insert(0, 'nombre_jugador', '')
        
        # Configurar dorsal como índice
        df_stats_indexed = df_stats.set_index('dorsal')
        
        # Editor de datos
        df_editado = st.data_editor(
            df_stats_indexed,
            use_container_width=True,
            column_config={
                "nombre_jugador": st.column_config.TextColumn(
                    "Nombre del Jugador",
                    help="Introduce el nombre del jugador"
                )
            },
            key=f"editor_stats_{equipo}"
        )

        # Descargar como CSV
        df_descarga = df_editado.reset_index()
        st.download_button(
            label="📥 Descargar estadísticas",
            data=df_descarga.to_csv(index=False).encode('utf-8'),
            file_name=f"estadisticas_{nombre_equipo.replace(' ', '_').lower()}.csv",
            mime='text/csv',
            key=f"download_{equipo}"
        )

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