from .utils import *
from ultralytics import YOLO
from inference import Tracker
from inference import Predictor
import supervision as sv

def main():

    # Cargar modelo YOLO
    model = YOLO("models/yolo9-60ep-8b-960imgsz.pt")  

    #Le voy a coger los frames sin predecir para no predecirlos todos, si no uno de cada 20, y luego aplicar un tracker, todo esto en la clase tracker
    predictor = Predictor(model)

    frames = read_video("/data/video_test/prueba3.mp4")

    tracker = Tracker(frames,model)
   
   
    
   


