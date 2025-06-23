import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import time
# Obtener el directorio raíz del proyecto de manera más robusta
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent  # Ajusta según tu estructura

# Agregar al path
sys.path.append(str(project_root))

MODELS_DIR = project_root / "models"

from utils import read_video, write_video, stubs_utils
from ultralytics import YOLO
from inference import Tracker, KeyPointer, MatchStats
from entities import Team, Match


# Aquí importarías tus módulos reales
# from inference.detector import PlayerDetector
# from utils.video_processor import VideoProcessor

class VideoUploader:
    """Clase para manejar la carga de videos"""
    
    def render_upload_section(self):
        """Renderizar la sección de carga de video"""
        st.markdown('<p class="section-header">📹 Carga de Video</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Selecciona un video para analizar",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Formatos soportados: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Video cargado: {uploaded_file.name}")
            
        return uploaded_file
    
    def show_original_video(self, uploaded_file):
        """Mostrar el video original"""
        st.subheader("Video Original")
        st.video(uploaded_file)
    
    def save_temp_file(self, uploaded_file):
        """Guardar archivo temporal"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name

class VideoProcessor:
    """Clase para procesar videos con tu sistema de IA"""
    
    def __init__(self):
 
        team_1= Team(1)
        team_2 = Team(2)
        self.match = Match(team_1,team_2)

        # Definir rutas de modelos de manera absoluta
        
        KEYPOINTS_MODEL_PATH = MODELS_DIR / "keypoints-500ep-48b-640imgsz.pt"
        DETECTIONS_MODEL_PATH = MODELS_DIR / "yolo9-60ep-8b-960imgsz.pt"
        #KEYPOINTS
        self.model_keypoints = YOLO(str(KEYPOINTS_MODEL_PATH))  # Cargar el modelo YOLOv8-pose tuneado
        self.keypointer = KeyPointer(self.model_keypoints, self.match)

        # Cargar modelo YOLO
        self.model = YOLO(str(DETECTIONS_MODEL_PATH))  # Cargar el modelo YOLOv9 tuneado

        #Objeto Tracker y Estadísticas del partido
        self.match_stats = MatchStats(self.match)
        self.tracker = Tracker(self.model, self.match, self.match_stats)

        
    
    def process_video(self, video_path, config):
        """
        Procesar video con tu sistema de detección
        
        Args:
            video_path: Ruta al video temporal
            config: Configuración del sidebar
            
        Returns:
            tuple: (video_anotado_path, mapa_superior_path, estadisticas)
        """
        # Control de ejecución única
        video_key = f"processed_{hash(video_path)}"
        if video_key in st.session_state:
            return (st.session_state['video_anotado'], 
               st.session_state.get('mapa_superior'), 
               st.session_state.get('estadisticas_t1', {}),
               st.session_state.get('estadisticas_t2', {}))

        # Mostrar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('🔄 Inicializando modelo...')
        progress_bar.progress(10)
        
        frames = read_video(video_path)
        output_frames = []
        output_field_images = []

        for i in range(0, len(frames), 25):
                
            
            progress = float(((i+1)/25)/30)
            progress_bar.progress(progress)

            frame_batch = frames[i:i+25]
            '''
            #TRACKING

            status_text.text('🔄 Detectando jugadores...')

            tracks_by_frame = self.tracker.detect_n_track(frame_batch, batch_number=int(i/25))
            
            #ANOTACIÓN + STATS

            status_text.text('🔄 Anotando frames' )
            self.tracker.draw_tracks(frame_batch, tracks_by_frame)

            status_text.text('🔄 Obteniendo estadísticas')
            self.tracker.stats.draw_possession(self.match_stats.get_match_stats(tracks_by_frame), tracks_by_frame, frame_batch)
            
            #KEYPOINTS AND MAP

            status_text.text('🔄 Generando mapa ')
            field_images = self.keypointer.keypoints_main_function(frame_batch, tracks_by_frame)
            
            #ALMACENAJE DE FRAMES
            output_field_images= output_field_images + field_images
            '''
            output_frames = output_frames + frame_batch
        
        
        video_anotado_path = project_root / "web/results/video_anotado.mp4"
        print(video_anotado_path)
        st.session_state['video_anotado'] = video_anotado_path

        try:
            write_video(output_frames, video_anotado_path)
            
            # DIAGNÓSTICO DETALLADO
            print(f"🔍 Diagnóstico completo:")
            print(f"   - Ruta: {video_anotado_path}")
            print(f"   - Archivo existe: {os.path.exists(video_anotado_path)}")
            
            if os.path.exists(video_anotado_path):
                file_size = os.path.getsize(video_anotado_path)
                print(f"   - Tamaño archivo: {file_size} bytes")
                
                # Verificar permisos
                print(f"   - Readable: {os.access(video_anotado_path, os.R_OK)}")
                print(f"   - Writable: {os.access(video_anotado_path, os.W_OK)}")
                
                # Verificar con OpenCV directamente
                import cv2
                cap = cv2.VideoCapture(str(video_anotado_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                print(f"   - Frames en video: {frame_count}")
                print(f"   - FPS: {fps}")
                print(f"   - Frames procesados originalmente: {len(output_frames)}")
                
            time.sleep(0.5)
    
        except Exception as e:
            print(f"❌ Error completo: {str(e)}")
            import traceback
            traceback.print_exc()
    
        if os.path.exists(video_anotado_path) and os.path.getsize(video_anotado_path) > 0:
            st.session_state['video_anotado'] = video_anotado_path
        else:
            st.error("❌ Error: El video está vacío")
            return None, None, None
        
        mapa_superior_path = project_root / "web/results/mapa_superior.mp4"
        st.session_state['mapa_superior'] = mapa_superior_path
        '''

        write_video(output_field_images,mapa_superior_path)
        '''
        estadisticas_t1 = {
            'jugadores_detectados': 22,
            'precision_promedio': 0.94,
            'frames_totales': 1500,
            'tiempo_procesamiento': 45.2,
            'detecciones_por_frame': 18.5,
            'equipo_local': 11,
            'equipo_visitante': 11,
            'confianza_promedio': config['confidence_threshold']
        }
        st.session_state['estadisticas_t1'] = estadisticas_t1
        
        estadisticas_t2 = {
            'jugadores_detectados': 22,
            'precision_promedio': 0.94,
            'frames_totales': 1500,
            'tiempo_procesamiento': 45.2,
            'detecciones_por_frame': 18.5,
            'equipo_local': 11,
            'equipo_visitante': 11,
            'confianza_promedio': config['confidence_threshold']
        }
        st.session_state['estadisticas_t2'] = estadisticas_t2
        status_text.text('✅ Procesamiento completado!')

        # Al final, antes del return:
        st.session_state[video_key] = True

        return video_anotado_path, mapa_superior_path, estadisticas_t1, estadisticas_t1
    
    def cleanup_temp_files(self, *file_paths):
        """Limpiar archivos temporales"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.warning(f"No se pudo eliminar archivo temporal: {e}")