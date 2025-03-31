import argparse as ap
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from filterpy.kalman import KalmanFilter
from sort import Sort

class Tracker:

    def __init__(self,frames,model):
        self.frames = frames #frames sin predecir
        self.detections=[]
        self.model = model
        self.tracker = Sort()

    def detect_frame(self,frame):

        #Detecta objetos en un frame y devuelve las detecciones en formato supervision 

        results = self.model.predict(frame)
        detections = sv.Detections.from_ultralytics(results)
        return detections
    
    def get_team_color(self, track_id):

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Diferentes colores
        return colors[track_id % len(colors)]  # Asigna colores cíclicamente


    def draw_player_marker(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2, track_id = track  # Asegúrate de que SORT devuelva track_id

            # Coordenadas del centro a los pies del jugador
            center_x = int((x1 + x2) / 2)
            center_y = int(y2)  # Parte inferior del bbox

            # Tamaño del óvalo
            width = int((x2 - x1) * 0.6)  # Ancho basado en el bbox
            height = int((y2 - y1) * 0.2)  # Alto pequeño para que parezca un marcador a los pies

            # Obtener color según el equipo
            color = self.get_team_color(track_id)

            # Dibujar el óvalo en la imagen
            cv2.ellipse(frame, (center_x, center_y), (width, height), 0, 0, 360, color, -1)

        return frame
    
    def detect_n_track(self):

        skip_frames = 20
        frame_count = 0

        for frame in self.frames:
            if frame_count % skip_frames == 0:

                # Inferencia con YOLO cada 20 frames
                sv_detections = self.detect_frame(frame)

                # Aplicar tracking
                dets = []
                for i in range(len(sv_detections.xyxy)):
                    x1, y1, x2, y2 = sv_detections.xyxy[i]
                    conf = sv_detections.confidence[i]
                    dets.append([x1, y1, x2, y2, conf])  # 1 = Score ficticio para el tracker

                dets = np.array(dets)
                # Aplicar tracker
                tracked_objects = self.tracker.update(dets)

            else:
               
                # Usar el tracker en frames intermedios
                tracked_objects = self.tracker.update(np.empty((0, 5)))  # No pasamos nuevas detecciones

            frame_count += 1
