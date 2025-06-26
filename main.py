from utils import read_video, write_video, stubs_utils
from ultralytics import YOLO
from inference import Tracker, KeyPointer, MatchStats
from entities import Team, Match
import supervision as sv
import torch
import os
import time

print("GPU disponible:", torch.cuda.is_available())
print("Inicio del programa")

def main():

    start_time = time.time()

    team_1= Team(1)
    team_2 = Team(2)
    match = Match(team_1,team_2)

    #Obtener frames del video
    frames = read_video("data/video_test/prueba3.mp4")

    #KEYPOINTS
    model_keypoints = YOLO("models/keypoints-500ep-48b-640imgsz.pt")  # Cargar el modelo YOLOv8-pose tuneado
    keypointer = KeyPointer(model_keypoints, match)

    # Cargar modelo YOLO
    model = YOLO("models/yolo9-60ep-8b-960imgsz.pt")  # Cargar el modelo YOLOv9 tuneado

    #Objeto Tracker y Estadísticas del partido
    match_stats = MatchStats(match)
    tracker = Tracker(model,match,match_stats)
    

    output_frames = []
    output_field_images = []
    all_tracks = []
    output_path = "stubs/prueba3.json"

    #CON STUBS
    if os.path.exists(output_path):
            print(f"Archivo {output_path} ya existe. Cargando datos guardados...")
            stub_tracks = stubs_utils.load_batches_from_json(output_path)

            for i in range(0, len(frames), 25):
            
                frame_batch = frames[i:i+25]

                #TRACKING
                tracks_by_frame = tracker.read_n_track(stub_tracks, batch_number=int(i/25), frame_batch= frame_batch)

                tracker.draw_tracks(frame_batch, tracks_by_frame)
                tracker.stats.draw_possession(match_stats.get_match_stats(tracks_by_frame), tracks_by_frame, frame_batch)

                #KEYPOINTS AND MAP
                field_images = keypointer.keypoints_main_function(frame_batch, tracks_by_frame)
                
                output_field_images = output_field_images + field_images
                output_frames = output_frames + frame_batch

    #CON PREDICCIÓN DEL MODELO
    else:
          
        for i in range(0, len(frames), 25):
                
            frame_batch = frames[i:i+25]
            detection_start_time = time.time()
            #TRACKING
            tracks_by_frame = tracker.detect_n_track(frame_batch, batch_number=int(i/25))
            all_tracks.append(tracks_by_frame)

            detection_end_time = time.time()
            detection_elapsed_time = detection_end_time - detection_start_time
            match_stats.total_detection_time += detection_elapsed_time
            print(f"Detection + asginación de equipos time en batch {(i/25)}: {detection_elapsed_time} ")

            annotation_start_time = time.time()
            tracker.draw_tracks(frame_batch, tracks_by_frame)
            tracker.stats.draw_possession(match_stats.get_match_stats(tracks_by_frame), tracks_by_frame, frame_batch)
            
            annotation_end_time = time.time()
            annotation_elapsed_time = annotation_end_time - annotation_start_time
            print(f"Annotation time en batch {(i/25)}: {annotation_elapsed_time} ")

            keypoints_start_time = time.time()
            #KEYPOINTS AND MAP
            field_images = keypointer.keypoints_main_function(frame_batch, tracks_by_frame)
            keypoint_end_time = time.time()
            keypoint_elapsed_time = keypoint_end_time - keypoints_start_time
            match_stats.total_keypoint_time += keypoint_elapsed_time
            print(f"Keypoints time en batch {(i/25)}: {keypoint_elapsed_time} ")

            output_field_images = output_field_images + field_images
            output_frames = output_frames + frame_batch
            
        stubs_utils.save_batches_to_json(all_tracks,output_path)

    stats_start = time.time()
    estadisticas_t1 = match.team_1.get_players_stats_sheets()
    estadisticas_t2 = match.team_2.get_players_stats_sheets()
    estadisticas_deteccion = match_stats.get_total_stats()
    stats_end = time.time()

    stats_time = stats_end-stats_start #prácticamente 0
    print(f"STATS_TIME = {stats_time}")
    #imprimimos las estadisiticas individuales
    print ("ESTADÍSTICAS EQUIPO 1:")
    match.team_1.print_players_stats()
    print ("ESTADÍSTICAS EQUIPO 2:")
    match.team_2.print_players_stats()

    print("ESTADÍSTICAS DEL PARTIDO")
    pos1, pos2 = tracker.stats.get_match_possession()
    print(f"Posesión equipo 1: {pos1} Posesión equipo 2: {pos2}")
    # Output: 45 55
    

    #Rearmar video
    write_video(output_frames, "data/video_test/prueba3_tracked_dorsal_rewrite.mp4")
    write_video(output_field_images, "data/video_test/prueba3_rewrite.mp4")
    print("Video guardado")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tiempo de ejecución: {elapsed_time:.2f} segundos")
    
if __name__ == "__main__":
    main()


