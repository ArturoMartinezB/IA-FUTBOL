import supervision as sv
import numpy as np
from utils import drawing_utils, get_color
from team import Team
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

        return tracks_by_frame
    

    def get_colors(self, frames, tracks_by_frame):
        
        players_colors = {}

        for num, frame in enumerate(frames):

            for tuple in tracks_by_frame[num]['players']:

                track_id = tuple[0]
                
                if track_id in players_colors or ( track_id in self.match.team_1.players or track_id in self.match.team_2.players):
                    pass

                else:

                    bbox = tuple[1]
                    x1,y1,x2,y2 = bbox
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Recorta el área del bounding box
                    cropped_bbox = frame[y1:y2, x1:x2]

                    color = get_color.get_color_player(cropped_bbox)

                    players_colors[track_id] = color

            
        return players_colors

    def assign_teams(self,players_colors):

        if players_colors is None:
            pass

        else:

            teams_colors , classified_players = get_color.get_teams_colors(players_colors)

            team1 , team2 = Team()

            team1.assign_team_color = teams_colors[0]
            team2.assign_team_color = teams_colors[0]

            for track_id, team in classified_players:

                if team == 0:
                    team1.add_player(track_id)
                
                else: 
                    team2.add_player(track_id)
            
            self.match.set_team_1(team1)
            self.match.set_team_2(team2)
                
    def check_new_players(self, track_by_frame):

        new_players = {}

        # Unir las claves de ambos diccionarios
        classified_players = set(self.match.team_1.keys()) | set(self.match.team_2.keys())

        for frame_number in enumerate(track_by_frame):
            # Extraer la lista de jugadores del frame actual
            unclassified = [
                (track_id, bbox)
                for (track_id, bbox) in track_by_frame[frame_number]['players']
                if track_id not in classified_players
            ]

            # Solo guardar si hay nuevos jugadores
            if unclassified:
                new_players[frame_number] = unclassified
            
        return new_players


    def assign_new_players(self, new_players, frames):

        for num, frame in enumerate(frames):

            for player in new_players[num]:

                track_id = player[0]

                bbox = player[1]
                x1,y1,x2,y2 = bbox
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Recorta el área del bounding box
                cropped_bbox = frame[y1:y2, x1:x2]

                color_jugador = get_color.get_color_player(cropped_bbox)

                dist1 = np.linalg.norm(color_jugador - self.match.team_1.get_team_color())
                dist2 = np.linalg.norm(color_jugador - self.match.team_2.get_team_color())

                if dist1 < dist2: 
                    self.match.team_1.add_player(track_id)
                
                else:
                    self.match.team_2.add_player(track_id)

    


    def draw_tracks(self, frames, tracks_by_frame):

        for num_frame, frame in enumerate(frames): 

            tracks = tracks_by_frame[num_frame]

            for player in tracks['players']:

                bbox = player[1]
                track_id = player[0]

                if self.match.team_1.belongs_here(track_id):

                    color = self.match.team_1.get_color()
                
                else:
                    color = self.match.team_2.get_color()
                

                drawing_utils.draw_ellipse(frame,color,bbox)
                drawing_utils.draw_banner(frame, color, bbox, track_id) #Esto habría que cambiarlo de sitio 
            
            #Formáto de color (BLUE, GREEN, RED)
            for ref in tracks['referees']:

                color = (255, 255, 0) # Los árbitros tienen el halo siempre de amarillo
                track_id = ref[0]
                bbox = ref[1]
                drawing_utils.draw_ellipse(frame,color, bbox)

            for gk in tracks['goalkeepers']:

                bbox = gk[1]
                color = (50, 50, 50) #De momento para los porteros gris
                track_id = gk[0]

                drawing_utils.draw_ellipse(frame, color, bbox)
                drawing_utils.draw_banner(frame,color,bbox,track_id)
            
            for ball in tracks['ball']:

                # Podría dibujar el pointer del color del equipo que tenga la posesión
                #color = match.get_team_in_possession
                color = ()
                bbox = ball[1]
                track_id = ball[0]

                drawing_utils.draw_ball_pointer(frame, bbox, track_id)
                
    
        


    def detect_n_track(self, frames):
        
        detections = []
        initial = True 

        for i in range(0, len(frames), 25):
            #hilo 1
            frame_batch = frames[i:i+25]
            print(f"Procesando batch {i//25} con {len(frame_batch)} frames")
            detections = self.detect_frames(frame_batch)
           
            #hilo 2
            tracks_by_frame = self.get_tracks(detections) # Devuelve un diccionario de 0 a 24 q contiene otros diccionarios diccionarios q son player, referee, goalkeeper, ball

            if initial:
                self.assign_teams(self.get_colors(frame_batch,tracks_by_frame)) # Obtengo los colores de los jugadores que aparecen en algún frame de los 25 y asigno los equipos 
                initial = False

            else:
                if (new_players := self.check_new_players(tracks_by_frame)):
                    self.assign_new_players(new_players, frame_batch)

            output_frames = self.draw_tracks(frame_batch,tracks_by_frame)

            break
     
        return output_frames
        
