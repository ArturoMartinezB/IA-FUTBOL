from utils import read_video, write_video
from ultralytics import YOLO
from inference import Tracker, KeyPointer
from entities import Team, Match
import supervision as sv
import torch
print("GPU disponible:", torch.cuda.is_available())

print("Inicio del programa")

def main():


    #Obtener frames del video
    frames = read_video("data/video_test/prueba3.mp4")



    #Introducir datos de los equipos para definir el partido
    '''print("Introduzca los datos del equipo 1")
    team_name_1 = input("Nombre: ")

    print("Introduzca los datos del equipo 2")
    team_name_2 = input("Nombre: ")

    
    #PrueBA KEYPOINTS
    
    model_keypoints = YOLO("models/keypoints-500ep-48b-640imgsz.pt")  # Cargar el modelo YOLOv8-pose tuneado
    #model_keypoints = YOLO("models/keypoints-80ep-16b-960imgsz.pt")
    keypointer = KeyPointer(model_keypoints)

    keypointer.get_keypoints(frames)
    '''
    team_1= Team('Bayern',1)
    team_2 = Team('Wolfsburg',2)
    match = Match(team_1,team_2)

    

    # Cargar modelo YOLO
    model = YOLO("models/yolo9-60ep-8b-960imgsz.pt")  # Cargar el modelo YOLOv9 tuneado

    
    #Trackear y dibujar
    tracker = Tracker(model,match)
    tracked_frames = tracker.detect_n_track(frames)
    
    #imprimimos las estadisiticas individuales
    print ("ESTADÍSTICAS EQUIPO 1:")
    match.team_1.print_players_stats()
    print ("ESTADÍSTICAS EQUIPO 2:")
    match.team_2.print_players_stats()
    
    #Rearmar video
    write_video(tracked_frames, "data/video_test/prueba3_tracked_dorsalX.mp4")
    print("Video guardado")
    
    
if __name__ == "__main__":
    main()


