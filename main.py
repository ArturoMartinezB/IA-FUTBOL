from utils import read_video, write_video
from ultralytics import YOLO
from inference import Tracker


import supervision as sv

print("Inicio del programa")

def main():

    # Cargar modelo YOLO
    model = YOLO("models/yolo9-60ep-8b-960imgsz.pt")  # Cargar el modelo YOLOv8 preentrenado

    frames = read_video("/data/video_test/prueba3.mp4")

    tracker = Tracker(model)
    print("Frames obtenidos y tracker inicializdo")
    tracked_frames = tracker.detect_n_track(frames)   
    write_video(tracked_frames, "/data/video_test/prueba3_tracked.mp4")
    
if __name__ == "__main__":
    main()


