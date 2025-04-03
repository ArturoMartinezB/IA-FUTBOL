import supervision as sv
#from sort.sort import Sort

class Tracker:

    def __init__(self,model):
        self.model = model
        self.tracker = sv.ByteTrack()
        self.clases = {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}
        self.track_order = {'bbox': 0, 'mask': 1, 'confidence': 2, 'class_id': 3, 'track_id': 4, 'class_name': 5} # orden de los elementos en el array de detecciones


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

    def get_tracks(self, detections):
        
        tracked_batch =[] 
        tracks_by_frame = {}
        
        for frame, dets in enumerate(detections):     #dets contiene las detecciones de un frame 

            #print ("DETS = ", dets)
            #FORMATO TRACKED_BATCH --> igual que detections pero con el track_id añadido 
            tracked_batch.append(self.tracker.update_with_detections(dets))

            players =[] # lista de tuplas de jugadores (track_id, xyxy)
            referees = []
            goalkeepers = []
            ball = []
            
           
            #print("Tracked_batch [-1]= ",tracked_batch[-1])
            tracked_frame = tracked_batch[-1] # última detección del frame actual
            print(tracked_frame)
            
            for tracks in tracked_frame:
                
                class_id = self.track_order['class_id']
                track_id = self.track_order['track_id']
                bbox = self.track_order['bbox']

                if tracks[class_id] == self.clases['player']:
                    players.append((tracks[track_id], tracks[bbox]))

                elif tracks[class_id] == self.clases['referee']:
                    referees.append((tracks[track_id], tracks[bbox]))

                elif tracks[class_id] == self.clases['goalkeeper']:  
                    goalkeepers.append((tracks[track_id], tracks[bbox]))

                else:
                    ball.append((tracks[track_id], tracks[bbox]))
            
            tracks_by_frame[frame] = {
                'players':players, 
                'referees':referees,
                'goalkeepers': goalkeepers,
                'ball': ball    
            }

            ''' dibujar_jugadores(players, frame)
            dibujar_arbitros(referees, frame) 
            señalizar_balón(ball, frame)
            o por otro lado
            dibujar_elemento(actores, frame)'''

        return tracks_by_frame
    

    def draw_tracks(self, frames, tracks_by_frame):


        pass


    def detect_n_track(self, frames):
        
        detections = []

        for i in range(0, len(frames), 25):
            #hilo 1
            frame_batch = frames[i:i+25]
            print(f"Procesando batch {i//25} con {len(frame_batch)} frames")
            detections = self.detect_frames(frame_batch)
           
            #hilo 2
            tracks_by_frame = self.get_tracks(detections) # Devuelve un diccionario de 0 a 24 q contiene otros diccionarios diccionarios q son player, referee, goalkeeper, ball
            output_frames = self.draw_tracks(frame_batch,tracks_by_frame)

            break
     
        return output_frames
        
