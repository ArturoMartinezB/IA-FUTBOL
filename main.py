from utils import read_video, write_video
from ultralytics import YOLO
from inference import Tracker, Match, Team
import supervision as sv
import torch
print("GPU disponible:", torch.cuda.is_available())

print("Inicio del programa")

def main():

    #Introducir datos de los equipos para definir el partido
    print("Introduzca los datos del equipo 1")
    team_name_1 = input("Nombre: ")
    team_color_1 = input("Color: ")

    print("Introduzca los datos del equipo 2")
    team_name_2 = input("Nombre: ")
    team_color_2 = input("Color: ")

    team_1= Team(team_name_1,team_color_1)
    team_2 = Team(team_name_2,team_color_2)

    match = Match(team_1,team_2)

    # Cargar modelo YOLO
    model = YOLO("models/yolo9-60ep-8b-960imgsz.pt")  # Cargar el modelo YOLOv8 preentrenado

    #Obtener frames del video
    frames = read_video("data/video_test/prueba3.mp4")

    #Trackear y dibujar
    tracker = Tracker(model,match)
    tracked_frames = tracker.detect_n_track(frames)

    #Rearmar video
    write_video(tracked_frames, "data/video_test/prueba3_tracked.mp4")
    print("Video guardado")

    
if __name__ == "__main__":
    main()


