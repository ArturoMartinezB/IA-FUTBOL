import numpy as np
import supervision as sv
#from sort.sort import Sort

class Tracker:

    def __init__(self,model):
        self.model = model
        self.tracker = sv.ByteTrack()
        self.clases = {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}



    #   Detecta objetos en varios frames y devuelve las detecciones en formato supervision 

    def detect_frames(self,frames):

        #prediciciones con el modelo
        results = self.model.predict(frames,device="cuda")

        ''' Se pasan los resultados a formato sv detections
            Formato Detections[[bbox][mask][confidence][clase d los obj detectados num][track_id][{'class_name': array('player', ...)}]] 
             El último array contiene el nombre de las clases de los objetos detectados'''
        
            
        #Convertir cada resultado a Detections y mantener el orden con una lista detections[0] = resultados del frame 0    
        detections = [sv.Detections.from_ultralytics(res) for res in results]

        #print(detections[0])
        return detections

    def update_detections(self, detections):
        
        tracked_batch =[]

        for dets in detections:

            #FORMTATO TRACKED_BATCH --> igual que detections pero con el track_id añadido 
            tracked_batch += self.tracker.update_with_detections(dets)

            players =[] # array de tuplas de jugadores (track_id, xyxy)
            referees = []
            goalkeepers = []
            ball = []

            tracked_frame = tracked_batch[-1]

            for index in range(len(tracked_frame.track_ids)):
                '''
                if tracked_frame.class_ids[index] == self.clases['player']:
                    players[tracked_frame.track_id[index]] =tracked_frame.xyxy[index]
                elif tracked_frame.class_ids[index] == self.clases['referee']:                      aquí lo hice con un diccionario, en este caso, separo los actores por tipo 
                    referees[tracked_frame.track_id[index]] =tracked_frame.xyxy[index]               pero necesito la lista de track_id además
                elif tracked_frame.class_ids[index] == self.clases['goalkeeper']:
                    goalkeepers[tracked_frame.track_id[index]] =tracked_frame.xyxy[index]
                else:
                    ball[tracked_frame.track_id[index]] =tracked_frame.xyxy[index]
            
            paquete = [tracked_frame.track_ids, players, referees, goalkeepers, ball] 
            
                '''
                if tracked_frame.class_ids[index] == self.clases['player']:
                    players.append(tracked_frame.track_ids[index], tracked_frame.xyxy[index])
                elif tracked_frame.class_ids[index] == self.clases['referee']:
                    referees.append(tracked_frame.track_ids[index], tracked_frame.xyxy[index])
                elif tracked_frame.class_ids[index] == self.clases['goalkeeper']:  
                    goalkeepers.append(tracked_frame.track_ids[index], tracked_frame.xyxy[index])
                else:
                    ball.append(tracked_frame.track_ids[index], tracked_frame.xyxy[index])
            
            #actores = [players, referees, goalkeepers, ball]

            ''' dibujar_jugadores(players, frame)
            dibujar_arbitros(referees, frame)
            señalizar_balón(ball, frame)
            o por otro lado
            dibujar_elemento(actores, frame)'''


        return tracked_batch

    def detect_n_track(self, frames):
        
        detections = []
        for i in range(0, len(frames), 25):
            #hilo 1
            frame_batch = frames[i:i+25]
            print(f"Procesando batch {i//25} con {len(frame_batch)} frames")
            detections = self.detect_frames(frame_batch)
           
            #hilo 2
            tracked_batch = self.update_detections(detections)
            #print(tracked_batch)

                            
        return detections[0]
        
