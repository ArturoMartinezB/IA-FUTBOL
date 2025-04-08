import supervision as sv
from utils import drawing_utils, get_color
#from sort.sort import Sort

class Tracker:

    def __init__(self,model,match):
        self.model = model
        self.tracker = sv.ByteTrack()
        self.match = match
        self.clases = {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}
        self.track_order = {'bbox': 0, 'mask': 1, 'confidence': 2, 'class_id': 3, 'track_id': 4, 'class_name': 5} # orden de los elementos en el array de detecciones


    #   Detecta objetos en varios frames y devuelve las detecciones en formato supervision 

    def detect_frames(self,frames):

        #Prediciciones con el modelo entrenado
        results = self.model.predict(frames,device="cuda")

        ''' Se pasan los resultados a formato sv detections ||  El último array contiene el nombre de las clases de los objetos detectados'''
            
        #Convertir cada resultado a Detections y mantener el orden con una lista detections[0] = resultados del frame 0    
        detections = [sv.Detections.from_ultralytics(res) for res in results]

        return detections

    def get_tracks(self, detections):
        
        tracked_batch =[] 
        tracks_by_frame = {}
        
        for frame, dets in enumerate(detections):     #dets contiene las detecciones de un frame 

            #FORMATO TRACKED_BATCH --> igual que detections pero con el track_id obtenido
            # Se actualiza el tracker y se guarda el frame trackeado (contiene las detections con los track_id)
            tracked_batch.append(self.tracker.update_with_detections(dets))
            

            # listas de tuplas (track_id, xyxy)
            players =[] 
            referees = []
            goalkeepers = []
            ball = []
            
            # frame recién trackeado
            tracked_frame = tracked_batch[-1]  
            
            #Recorremos todos los objetos trackeados y según la clase se guarda (track_id, bbox) en la lista correspondiente
            #Después las listas se añaden a un diccionario con key = nº frame (0-24)
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

    def assign_teams(self, frames, tracks_by_frame):

        for num, frame in enumerate(frames):

            player_colors = []
            for tuple in tracks_by_frame[num]['players']:

                track_id = tuple[0]

                bbox = tuple[1]
                x1,y1,x2,y2 = bbox
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Recorta el área del bounding box
                cropped_bbox = frame[y1:y2, x1:x2]

                color = get_color.get_color_player(cropped_bbox)

                player_colors[num] = [track_id, bbox, color]

                if num == 0:
                    teams_colors = get_color.get_teams_colors
                
                




        


    def draw_tracks(self, frames, tracks_by_frame):

        for num_frame, frame in enumerate(frames): 

            tracks = tracks_by_frame[num_frame]

            for player in tracks['players']:

                bbox = player[1]
                #color = ____.getcolor(frame, bbox)
                track_id = player[0]
                

                drawing_utils.draw_ellipse(frame,color,bbox)
                drawing_utils.draw_banner(frame, color, bbox, track_id) #Esto habría que cambiarlo de sitio 
            
            for ref in tracks['referees']:

                color = (255, 255, 0) # Los árbitros tienen el halo siempre de amarillo
                track_id = ref[0]
                bbox = ref[1]
                drawing_utils.draw_ellipse(frame,color, bbox)

            for gk in tracks['goalkeepers']:

                bbox = gk[1]
                color = None #** Debo asignar un color a los porteros.
                track_id = gk[0]

                drawing_utils.draw_ellipse(frame, color, bbox)
                drawing_utils.draw_banner(frame,color,bbox,track_id)
            
            for ball in tracks['ball']:

                # Podría dibujar el pointer del color del equipo que tenga la posesión
                #color = match.get_team_in_possession
                bbox = ball[1]
                track_id = ball[0]

                drawing_utils.draw_ball_pointer(frame, bbox, track_id)
                

        


    def detect_n_track(self, frames):
        
        detections = []

        for i in range(0, len(frames), 25):
            #hilo 1
            frame_batch = frames[i:i+25]
            print(f"Procesando batch {i//25} con {len(frame_batch)} frames")
            detections = self.detect_frames(frame_batch)
           
            #hilo 2
            tracks_by_frame = self.get_tracks(detections) # Devuelve un diccionario de 0 a 24 q contiene otros diccionarios diccionarios q son player, referee, goalkeeper, ball
            self.assign_teams(frame_batch,tracks_by_frame)
            output_frames = self.draw_tracks(frame_batch,tracks_by_frame)

            break
     
        return output_frames
        
