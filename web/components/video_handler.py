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


        # Control de ejecución única
        video_key = f"processed_{hash(video_path)}"
        if video_key in st.session_state:
            return (st.session_state['video_anotado'], 
               st.session_state.get('mapa_superior'),
               st.session_state.get('estadisticas_proceso', {}), 
               st.session_state.get('estadisticas_t1', {}),
               st.session_state.get('estadisticas_t2', {}))
        

        resultados = config['resultados']
        video_anotado_selected = resultados['Video anotado']
        mapa_superior_selected = resultados['Mapeado del video']
        stats_1_selected = resultados['Estadísticas equipo 1']
        stats_2_selected = resultados['Estadísticas equipo 2']
        stats_process_selected = resultados['Estadísticas del procesamiento']


        # Mostrar progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text('🔄 Inicializando modelo...')
        
        frames = read_video(video_path)
        output_frames = []
        output_field_images = []
        
        for i in range(0, len(frames), 25):
                
            
            progress = float(((i+1)/25)/30)
            progress_bar.progress(progress)

            frame_batch = frames[i:i+25]
            
            #TRACKING

            status_text.text('🔄 Detectando jugadores...')

            detection_start= time.time()

            tracks_by_frame = self.tracker.detect_n_track(frame_batch, batch_number=int(i/25))

            detection_end= time.time()
            detection_time = detection_end - detection_start
            self.match_stats.total_detection_time += detection_time

            #ANOTACIÓN + STATS

            
            status_text.text('🔄 Anotando frames' )
            self.tracker.draw_tracks(frame_batch, tracks_by_frame)

            
            status_text.text('🔄 Obteniendo estadísticas')
            self.tracker.stats.draw_possession(self.match_stats.get_match_stats(tracks_by_frame), tracks_by_frame, frame_batch)
            
            #KEYPOINTS AND MAP
            
            status_text.text('🔄 Generando mapa ')

            keypoint_start = time.time()

            field_images = self.keypointer.keypoints_main_function(frame_batch, tracks_by_frame)

            keypoint_end = time.time()
            keypoint_time = keypoint_end - keypoint_start
            self.match_stats.total_keypoint_time += keypoint_time
            output_field_images = output_field_images + field_images
            
            #ALMACENAJE DE FRAMES
            
            output_frames = output_frames + frame_batch
        
        
        #GUARDADO DE VÍDEOS Y RETURNS

        #Video anotado
        video_anotado_path = project_root / "web/results/video_anotado.mp4"
        st.session_state['video_anotado'] = video_anotado_path

        write_video(output_frames, video_anotado_path)
     
        #Mapa superior
        mapa_superior_path = project_root / "web/results/mapa_superior.mp4"
        st.session_state['mapa_superior'] = mapa_superior_path
    
        write_video(output_field_images,mapa_superior_path)

        #Estadísticas
        estadisticas_process = self.match_stats.get_total_stats()
        st.session_state['estadisticas_proceso'] = estadisticas_process


        #Estas no las pongo en función del bool porque no sabemos cual equipo es cual.
        estadisticas_t1 = self.match.team_1.get_players_stats_sheets()
        st.session_state['estadisticas_t1'] = estadisticas_t1
        estadisticas_t2 = self.match.team_2.get_players_stats_sheets()
        st.session_state['estadisticas_t2'] = estadisticas_t2

        

        status_text.text('✅ Procesamiento completado!')
        progress_bar.progress(100)

        # Al final, antes del return:
        st.session_state[video_key] = True

        return video_anotado_path, mapa_superior_path, estadisticas_process, estadisticas_t1, estadisticas_t2
    
    def cleanup_temp_files(self, *file_paths):
        """Limpiar archivos temporales"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.warning(f"No se pudo eliminar archivo temporal: {e}")