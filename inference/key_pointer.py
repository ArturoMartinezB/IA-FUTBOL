import supervision as sv
import os
import cv2
import numpy as np


class KeyPointer:

    def __init__(self,model):

        self.model = model
        

    def prediction (self, frames):

        #results =  #Prediciciones con el modelo entrenado
            self.model.predict(self.redimensionar_frames(frames), save=True , imgsz=640)

    def redimensionar_frames(self, frames, size=(640, 640)):
        frames_redimensionados = []
        for frame in frames:
            resized = cv2.resize(frame, size)  # Redimensiona a (width, height)
            frames_redimensionados.append(resized)
        return frames_redimensionados

    def get_keypoints(self,frames):

         
        for i in range(0, len(frames), 25):
            #hilo 1
            frame_batch = frames[i:i+25]
            print(f"Procesando batch {(i//25)} / 30 con {len(frame_batch)} frames")
             # Crea un directorio único para cada batch

            print(type(frame_batch[0]))
           
            self.prediction(frame_batch)

            
        

