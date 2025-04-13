import supervision as sv
import numpy as np
from utils import color_utils, drawing_utils
from inference.team import Team

class Tracker:

    def __init__(self,model,match):
        self.model = model
        self.tracker = sv.ByteTrack()
        self.match = match
        self.clases = {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referee': 3}
        self.track_order = {'bbox': 0, 'mask': 1, 'confidence': 2, 'class_id': 3, 'track_id': 4, 'class_name': 5} # orden de los elementos en el array de detecciones


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
    
    def get_players_colors(self, frames, tracks_by_frame):
        
        players_colors = {}

        for num, frame in enumerate(frames):
            
            for tuple in tracks_by_frame[num]['players']:
                track_id = int(tuple[0])
            
                bbox = tuple[1]
                x1, y1, x2, y2 = map(int, bbox)

                # Recorta el área del bounding box
                cropped_bbox = frame[y1:y2, x1:x2]

                # Obtiene el color y lo añade a la lista del track_id
                color = color_utils.get_color_player(cropped_bbox)

                #Guardo los colores por track_id obtenidos de distintos frames
                
                if track_id in players_colors:
                    actual_colors_list = players_colors[track_id]
                else:
                    actual_colors_list = []

                updated_list = [color] + actual_colors_list
                players_colors[track_id] = updated_list
            

        avg_colors = {}

        for track_id, color_list in players_colors.items():
            avg_color = np.mean(color_list, axis=0)
            avg_colors[track_id] = avg_color.astype(int).tolist()
            
        return avg_colors        #para que cuadre --> antes era players_colors[track_id]= color

    def assign_teams(self,players_colors):

        if players_colors is None:
            pass

        else:

            teams_colors , classified_players = color_utils.get_teams_colors(players_colors)

            self.match.team_1.assign_team_color(teams_colors[0])
            self.match.team_2.assign_team_color(teams_colors[1])

            for track_id, team in classified_players.items():

                if team == 0:
                    self.match.team_1.add_player(track_id)
                
                else: 
                    self.match.team_2.add_player(track_id)
                        
    def check_new_players(self, track_by_frame):

        new_players = {}

        # Unir las claves de ambos diccionarios
        classified_players = set((self.match.team_1.players.values())) | set(self.match.team_2.players.values())

        for frame_number , _ in enumerate(track_by_frame):
            # Extraer la lista de jugadores del frame actual
            unclassified = [
                (track_id, bbox)
                for (track_id, bbox) in track_by_frame[frame_number]['players']
                if track_id not in classified_players
            ]

            # Solo guardar si hay nuevos jugadores
            if unclassified:
                new_players[frame_number] = unclassified
            else: 
                new_players[frame_number] = []
            
        return new_players

    
    def recover_track_id(self, frame_number, player_id, bbox, track_by_frame, team):
        
        if team == 1:
            team = self.match.team_1
            other_team = self.match.team_2

        else: 
            team = self.match.team_2
            other_team = self.match.team_1

        # Extraer los track_id del frame
        track_ids_in_frame = {track_id for track_id, _  in track_by_frame[frame_number]['players']}

        #Ignorar los asignados al otro equipo
        other_team_ids = other_team.players.values()
        team_ids_in_frame = track_ids_in_frame - other_team_ids

        # Obtener los track_id asignados a un equipo pero que no aparecen en el campo ( han desaparecido y probablemente se les ha asignado el id que queremos recuperar)
        team_ids = set(team.players.values()) 
        lost_ids = team_ids - team_ids_in_frame

        if len(lost_ids) == 1:
            key = [k for k, value in team.players.items() if value == lost_ids[0]]

        elif len(lost_ids) > 1:
            #obtener el que tenga mínima distancia entre los track_id desaparecidos y la última posición detectada
            for lost in lost_ids:
                #Ahora tengo que calcular las distancias entre el player_id, bbox de los argumentos y los bboxes de los track_id lost, que los tengo en algún frame de track_by_frame 
                #(o no, en ese caso debería guardar en el diccionario player el último bbox detectado de cada jugador o algo similar)
                pass      
            pass  
        else: 
            print("Error al recuperar al jugador")

        '''
        for track_id, bbox_it in track_by_frame[frame_number]['players']:
            
            
            if track_id not in track_ids_in_team:
            
                
                x1, y1, x2, y2 = bbox_it
                center_it = (int((x1 + x2) / 2), int((y1+y2)/2))


                distance_it = abs(center-center_it)

                if distance_it < distance:
                    distance = distance_it
                    former_track_id = track_id

'''


    def assign_new_players(self, new_players, frames, tracks_by_frame):

        for num, frame in enumerate(frames):

            for player in new_players[num]:

                track_id = player[0]

                bbox = player[1]
                x1,y1,x2,y2 = bbox
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Recorta el área del bounding box
                cropped_bbox = frame[y1:y2, x1:x2]

                color_jugador = color_utils.get_color_player(cropped_bbox)

                dist1 = np.linalg.norm(color_jugador - self.match.team_1.color)
                dist2 = np.linalg.norm(color_jugador - self.match.team_2.color)

                if dist1 < dist2: 
                    if self.match.team_1.add_player(track_id) == False:
                        self.recover_track_id(num, track_id, bbox, tracks_by_frame, 1)
                        
                else:
                    if self.match.team_2.add_player(track_id) == False:
                        self.recover_track_id(num, track_id, bbox, tracks_by_frame, 2)


    def draw_tracks(self, frames, tracks_by_frame):

        for num_frame, frame in enumerate(frames): 

            tracks = tracks_by_frame[num_frame]

            for player in tracks['players']:

                bbox = player[1]
                track_id = player[0]

                if self.match.team_1.belongs_here(track_id):

                    color = self.match.team_1.color
                
                else:
                    color = self.match.team_2.color
                
                drawing_utils.draw_ellipse(frame, color, bbox)
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
                bbox = ball[1]
                track_id = ball[0]

                drawing_utils.draw_ball_pointer(frame, bbox)
                

    def detect_n_track(self, frames):
        
        detections = []
        initial = True 
        output_frames = []

        for i in range(0, len(frames), 25):
            #hilo 1
            frame_batch = frames[i:i+25]
            print(f"Procesando batch {i//25} / 30 con {len(frame_batch)} frames")
            detections = self.detect_frames(frame_batch)
           
            #hilo 2
            tracks_by_frame = self.get_tracks(detections) # Devuelve un diccionario de 0 a 24 q contiene otros diccionarios diccionarios q son player, referee, goalkeeper, ball

            if initial:
                self.assign_teams(self.get_players_colors(frame_batch,tracks_by_frame)) # Obtengo los colores de los jugadores que aparecen en algún frame de los 25 y asigno los equipos 
                initial = False

            else:
                if (new_players := self.check_new_players(tracks_by_frame)):
                    self.assign_new_players(new_players, frame_batch, tracks_by_frame)

            self.draw_tracks(frame_batch,tracks_by_frame)
            
            output_frames = output_frames+ frame_batch

        return output_frames
