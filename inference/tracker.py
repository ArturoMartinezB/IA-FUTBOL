
import argparse as ap
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from sort.sort import Sort
from sklearn.cluster import KMeans

class Tracker:

    def __init__(self,model):
        self.detections=[]
        self.model = model
        self.tracker = Sort()

    def detect_frame(self,frame):

        #Detecta objetos en un frame y devuelve las detecciones en formato supervision 

        results = self.model.predict(frame)
        detections = sv.Detections.from_ultralytics(results)
        return detections
    
    def get_team_color(self, frame,x1, y1, x2, y2,k = 3):
        
        print("Calculando color del equipo...")
        #Calcular ancho y alto
        w = x2 - x1
        h = y2 - y1

        # Extraer la mitad superior del bounding box
        half_h = h // 2
        roi = frame[y1:y1+half_h, x1:x2]
        
        # Convertir ROI en una lista de píxeles
        pixels = roi.reshape(-1, 3)
        
        # Aplicar K-Means
        try:
            kmeans = KMeans(n_clusters=k, n_init=10)
            labels = kmeans.fit_predict(pixels)
            
            # Contar frecuencia de cada cluster
            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_cluster]
        
            return tuple(map(int, dominant_color))  # Convertir a enteros
        
        except:
            # Si K-Means falla, calcular color promedio
            avg_color = np.mean(pixels, axis=0)
            return tuple(map(int, avg_color))



    def draw_player_marker(self, frame, tracked_objects):

        print("Drawing player markers...")
        for object in tracked_objects:
            x1, y1, x2, y2, track_id, clase = object  # Asegúrate de que SORT devuelva track_id
            
            if clase != 0: # Solo procesar jugadores (clase ?= 0)
                break

            # Coordenadas del centro a los pies del jugador
            center_x = int((x1 + x2) / 2)
            center_y = int(y2)  # Parte inferior del bbox

            # Tamaño del óvalo
            width = int((x2 - x1) * 0.6)  # Ancho basado en el bbox
            height = int((y2 - y1) * 0.2)  # Alto pequeño para que parezca un marcador a los pies

            # Obtener color según el equipo
            color = self.get_team_color(frame, x1, y1, x2, y2)

            # Dibujar el óvalo en la imagen
            cv2.ellipse(frame, (center_x, center_y), (width, height), 0, 0, 360, color, -1)

        print("Player markers drawn.")
        return frame
    
    def detect_n_track(self, frames):

        skip_frames = 20
        frame_count = 0
        tracked_frames = []

        print("Detectando y rastreando objetos...")
        print(len(frames))
        for frame in frames:
            print("punto 1")
            if frame_count % skip_frames == 0:
                
                # Inferencia con YOLO cada 20 frames
                sv_detections = self.detect_frame(frame)

                # Aplicar tracking
                dets = []
                for i in range(len(sv_detections.xyxy)):
                    x1, y1, x2, y2 = sv_detections.xyxy[i]
                    clase = sv_detections.class_id[i]
                    conf = sv_detections.confidence[i]
                    dets.append([x1, y1, x2, y2, conf,clase])  
                dets = np.array(dets)
                # Aplicar tracker
                tracked_objects = self.tracker.update(dets)
                
                
            else:
               
                # Usar el tracker en frames intermedios
                tracked_objects = self.tracker.update(np.empty((0, 5))) # No pasamos nuevas detecciones COMPROBAR EL SIZE DE ESTE ARRAY
                print("Frame sin predicciones")
            #pintar el frame
            tracked_frames.append(self.draw_player_marker(frame, tracked_objects)) #<-- atemporal, elementos trackeados continuamente
            frame_count += 1
        
        print("Detección y rastreo completados.")
        return tracked_frames
