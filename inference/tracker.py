import supervision as sv
import numpy as np
import os
from itertools import combinations
from utils import color_utils, drawing_utils, bbox_utils, stubs_utils
from inference.ball_interpolator import BallInterpolator
from inference.matchstats import MatchStats

from scipy.optimize import linear_sum_assignment


class Tracker:

    def __init__(self,model,match):
        self.model = model
        self.tracker = sv.ByteTrack()
        self.match = match
        self.stats =MatchStats(match)
        self.ball_interpolator = BallInterpolator()
        self.clases = {'ball': 0, 'goalkeeper': 1, 'player': 2, 'referees': 3}
        self.track_order = {'bbox': 0, 'mask': 1, 'confidence': 2, 'class_id': 3, 'track_id': 4, 'class_name': 5} # orden de los elementos en el array de detecciones

    def detect_frames(self,frames):

        #Prediciciones con el modelo entrenado
        results = self.model.predict(frames,device="cuda:1")

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
            for cont, tracks in enumerate(tracked_frame):

                #print("TRACK-",cont)
                class_id = self.track_order['class_id']
                track_id = self.track_order['track_id']
                bbox = self.track_order['bbox']

                if tracks[class_id] == self.clases['player']:
                    players.append((tracks[track_id], tracks[bbox]))

                elif tracks[class_id] == self.clases['referees']:
                    
                    referees.append((tracks[track_id], tracks[bbox]))

                elif tracks[class_id] == self.clases['goalkeeper']:
                
                    goalkeepers.append((tracks[track_id], tracks[bbox]))

                elif tracks[class_id] == self.clases['ball']:   
                    pass
            
            #AL TENER UN SCORE BAJO LA DETECCIÓN DEL BALÓN, EL TRACKER LO ESTÁ IGNORANDO
            # AÑADIRLO MANUALMENTE A TRACKS_BY_FRAME DESDE DETS

            for i, obj in enumerate(dets):
                bbox = obj[0].tolist()  
                class_id = int(obj[3])  

                if class_id == self.clases['ball']:

                    ball.append(("O", bbox))

            tracks_by_frame[frame] = {
                'players':players, 
                'referees':referees,
                'goalkeepers': goalkeepers,
                'ball': ball    
            }

        return tracks_by_frame
   
    def assign_teams(self,players_colors):

        if players_colors is None:
            pass

        else:

            teams_colors , classified_players = color_utils.get_teams_colors(players_colors)

            self.match.team_1.assign_team_color(teams_colors[0])
            self.match.team_2.assign_team_color(teams_colors[1])

            for track_id, team in classified_players.items():
                
                if team == 0:
                    if False == self.match.team_1.add_player(track_id):
                        print("no se ha añadido al jugador: ", track_id)
                        
                
                else: 
                    if False == self.match.team_2.add_player(track_id):
                        print("no se ha añadido al jugador_2: ", track_id)

    def assign_referees(self, tracks_by_frame):

        for frame_num, _ in enumerate(tracks_by_frame):
            if tracks_by_frame[frame_num]['referees']:
                for referee in tracks_by_frame[frame_num]['referees']:
                    self.match.add_referee(referee[0])             
                        
    def check_new_players(self, track_by_frame):

        new_players = {}

        # Unir los valores de ambos diccionarios
        classified_players1 = set(self.match.team_1.players.values()) 
        classified_players2 = set(self.match.team_2.players.values())

        for frame_number , _ in enumerate(track_by_frame):
            # Extraer la lista de jugadores del frame actual
            unclassified = []
            for (track_id, bbox) in track_by_frame[frame_number]['players']:
                if track_id not in classified_players1 | classified_players2 and track_id not in self.match.referees:
                    unclassified.append((track_id,bbox))

                elif track_id in classified_players1:
                    self.match.team_1.update_last_position(track_id,bbox)
                
                else:
                    self.match.team_2.update_last_position(track_id,bbox)
            
            # Solo guardar si hay nuevos jugadores
            if unclassified:
                new_players[frame_number] = unclassified
            else: 
                new_players[frame_number] = []
            
            #print ( "NUEVOS JUGADORES: ", new_players)
        return new_players

    def get_lost_ids(self,team_id, frame_number, track_by_frame):

        if team_id == 1:
            team = self.match.team_1
            other_team = self.match.team_2

        else: 
            team = self.match.team_2
            other_team = self.match.team_1

        # Extraer los track_id del frame
        track_ids_in_frame = {track_id for track_id, _  in track_by_frame[frame_number]['players']}

        #Ignorar los asignados al otro equipo
        other_team_ids = set(other_team.players.values())
        team_ids_in_frame = {int(tid) for tid in track_ids_in_frame - set(other_team_ids)}

        # Obtener los track_id asignados a un equipo pero que no aparecen en el campo ( han desaparecido y probablemente se les ha asignado el id que queremos recuperar)
        team_ids = set(team.players.values()) 
        #print(f"Team_ids in frame:", team_ids_in_frame)
        lost_ids = team_ids - team_ids_in_frame
        #print(f"Lost_ids frame[{frame_number}]: ",lost_ids)

        return lost_ids

    def get_key(self, dicc, value):

        key = [k for k, v in dicc.items() if v == value ]
        
        return  key[0]

    def recover_track_id(self, frame_number, player_id, bbox, track_by_frame, team_id):
        
        recovered = False

        if team_id == 1:
            team = self.match.team_1

        else: 
            team = self.match.team_2

        lost_ids = self.get_lost_ids(team_id,frame_number,track_by_frame)

        #ajustar formato
        player_id = int(player_id)
        players = team.players

        if len(lost_ids) == 1 and  next(iter(lost_ids)) is not None:

            lost_id = next(iter(lost_ids))

            key = self.get_key(players, lost_id)

            recovered = True

            if team_id == 1:
                self.match.team_1.players[key] = player_id

                #añado el cambio de track_id en las stats
                self.match.team_1.get_player_stats(key).add_track_id(player_id)
                #print(f"JUGADOR {player_id} añadido al equipo 1, con el dorsal {key}")
            else:
                self.match.team_2.players[key] = player_id

                #añado el cambio de track_id en las stats
                self.match.team_2.get_player_stats(key).add_track_id(player_id)

                #print(f"JUGADOR {player_id} añadido al equipo 2, con el dorsal {key}")

        elif len(lost_ids) > 1:
            #obtener el que tenga mínima distancia entre los track_id desaparecidos y la última posición detectada
            #if frame_number != 0:
            lost_players_with_bbox = []

            for i in range(1, frame_number):

                for player in track_by_frame[frame_number-i]['players']:
                    
                    for lost in lost_ids:

                        if player[0] == lost:
                            lost_players_with_bbox.append(player)
            
            if lost_players_with_bbox == []:

                for lost in lost_ids:
                    print("Lost _id linea 241 ", lost)
                    if lost is not None:
                        lost_players_with_bbox.append((lost, team.last_position[lost]))  # FALTA UN CASO POR CUBRIR que un track_id si que esté en este batch y el otro no 
                                                                                        #(dudo sobre la veracidad del último bbox)
            nearest_track_id = bbox_utils.nearest_bbox(bbox,lost_players_with_bbox)

            key = self.get_key(players, nearest_track_id)
                
            '''else: 
                print("EQUIPO 1:", self.match.team_1.players.values())
                print("EQUIPO 2:", self.match.team_2.players.values())
                print("Lost ids: ", lost_ids)
                nearest_track_id = team.last_position[int(list(lost_ids)[0])]
                print("Nearest track_id : ", nearest_track_id)
                key = self.get_key(players, nearest_track_id)
            '''
            recovered = True

            if team_id == 1:
                self.match.team_1.players[key] = int(player_id)
                #print(f"JUGADOR {player_id} añadido al equipo 1, con el dorsal {key}")
            else:
                self.match.team_2.players[key] = int(player_id)
                #print(f"JUGADOR {player_id} añadido al equipo 2, con el dorsal {key}")  

        else: 
            #print("Error de detección, falso positivo, track_id no incorporado: ", player_id)
            pass

        return recovered
        
    def assign_new_players(self, new_players, frames, tracks_by_frame):

        added = []
        for num, frame in enumerate(frames):

            if new_players.get(num):
                for player in new_players[num]:

                    track_id = player[0]
                    #para solo añadirlo al primer frame
                    if track_id in added:
                        break

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
                            print("Este se intenta añadir a través de recover al team 1: ", track_id)
                            if self.recover_track_id(num, track_id, bbox, tracks_by_frame, 1):
                                added.append(track_id)
                                print("se añadió con recover_track_id: ", track_id)
                                self.match.team_1.update_last_position(track_id,bbox)
                        else:
                            self.match.team_1.update_last_position(track_id,bbox)
                        
                            
                    else:
                        if self.match.team_2.add_player(track_id) == False:
                            print("Este se intenta añadir a través de recover al team 2: ", track_id)
                            if self.recover_track_id(num, track_id, bbox, tracks_by_frame, 2):
                                added.append(track_id)
                                self.match.team_2.update_last_position(track_id,bbox)
                        else:
                            self.match.team_2.update_last_position(track_id,bbox)
                
    def initial_positions(self,tracks_by_frame):

        for i in range(0,24):

            for track_id, bbox in tracks_by_frame[i]['players']:
                
                if track_id in set(self.match.team_1.players.values()):
                    self.match.team_1.update_last_position(track_id,bbox)

                elif track_id in set(self.match.team_2.players.values()):
                    self.match.team_2.update_last_position(track_id,bbox)

    def draw_tracks(self, frames, tracks_by_frame):

        for num_frame, frame in enumerate(frames): 

            tracks = tracks_by_frame[num_frame]

            for player in tracks['players']:

                bbox = player[1]
                track_id = player[0]

                if self.match.team_1.belongs_here(track_id):

                    color = self.match.team_1.color
                    dorsal = self.match.team_1.get_dorsal(track_id)
                
                elif self.match.team_2.belongs_here(track_id):
                    color = self.match.team_2.color
                    dorsal = self.match.team_2.get_dorsal(track_id)
                    
                else:
                    color = (0,125,225)
                    dorsal = None
                
                drawing_utils.draw_ellipse(frame, color, bbox)
                drawing_utils.draw_banner(frame, color, bbox, dorsal)
            
            #Formáto de color (BLUE, GREEN, RED)
            for ref in tracks['referees']:

                color = (255, 255, 0) # Los árbitros tienen el halo siempre de amarillo
                track_id = ref[0]
                bbox = ref[1]
                drawing_utils.draw_ellipse(frame,color, bbox)
                drawing_utils.draw_banner(frame, color, bbox, track_id)

            for gk in tracks['goalkeepers']:

                bbox = gk[1]
                color = (50, 50, 50) #De momento para los porteros gris
                track_id = gk[0]

                drawing_utils.draw_ellipse(frame, color, bbox)
                drawing_utils.draw_banner(frame,color,bbox,dorsal)
            
            for ball in tracks['ball']:

                # Podría dibujar el pointer del color del equipo que tenga la posesión
                #color = match.get_team_in_possession
                bbox = ball[1]
                track_id = ball[0]

                drawing_utils.draw_pointer(frame, bbox)
                
    def check_collision(self, frames, tracks_by_frame):

        collided_track_ids = []

        for frame_num in range(len(frames)):
            players = tracks_by_frame[frame_num]['players']
            for (track_id1, bbox1), (track_id2, bbox2) in combinations(players, 2):
                 # Evitar comparar jugadores del mismo equipo
                team1 = self.match.belongs_to(track_id1)
                team2 = self.match.belongs_to(track_id2)

                if team1 is not None and team2 is not None and team1 == team2:
                    continue  # están en el mismo equipo, no se comparan

                if bbox_utils.euclidean_distance(bbox1, bbox2) < 25:
                    
                    if (track_id1, track_id2) not in collided_track_ids and (track_id2, track_id1) not in collided_track_ids:
                        #print(f"Possible collision between {track_id1} and {track_id2} in frame {frame_num}")
                        collided_track_ids.append((track_id1, track_id2))


        return collided_track_ids
    
    def check_wrong_team_assignation(self, frames, tracks_by_frame):

        colors_per_id = color_utils.get_players_colors(frames,tracks_by_frame)
        wrongly_in_team1 = []
        wrongly_in_team2 = []

        for num, frame in enumerate(tracks_by_frame):
            for track_id , bbox in tracks_by_frame[num]['players']:

                    team_assigned_1 = self.match.belongs_to(track_id)  

                    distance_1_1 = color_utils.color_distance(colors_per_id[track_id], self.match.team_1.color)
                    distance_1_2 = color_utils.color_distance(colors_per_id[track_id], self.match.team_2.color)


                    if distance_1_1 < distance_1_2 and team_assigned_1 == 2: 
                        if not any(tid == track_id for tid, _ in wrongly_in_team1):
                            wrongly_in_team1.append((track_id,bbox))

                    elif distance_1_1 > distance_1_2 and team_assigned_1 == 1: 
                        if not any(tid == track_id for tid, _ in wrongly_in_team2):
                            wrongly_in_team2.append((track_id,bbox))

        if len(wrongly_in_team1) > 0 or len(wrongly_in_team2) > 0:

            if len(wrongly_in_team1) == len(wrongly_in_team2):
                
                #Para obtener la combinación de pares de jugadores que minimice las distancias entre bboxes, un erróneo de cada equipo, utilizo el Algoritmo de Asignación Húngaro
                cost_matrix = np.zeros((len(wrongly_in_team1), len(wrongly_in_team2)))

                for i, (_, bbox1) in enumerate(wrongly_in_team1):
                    for j, (_, bbox2) in enumerate(wrongly_in_team2):
                        cost_matrix[i, j] = bbox_utils.euclidean_distance(bbox1, bbox2)
                                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matches = [(wrongly_in_team1[i], wrongly_in_team2[j]) for i, j in zip(row_ind, col_ind)]

                for (track_id1, bbox1), (track_id2, bbox2) in matches:
                    
                    print(f"Se pretende swappear a {track_id1} con {track_id2}")
                    self.swap_players(track_id1,track_id2)


            else:

                print(f"Hay jugadores mal asignados asimétricamente, hay {len(wrongly_in_team1)} mal asignados en el equipo 1 y {len(wrongly_in_team2)} mal asignados en el equipo 2")

    def check_changed_team(self, collided_track_ids, colors_per_id): 

        if collided_track_ids:

            for track_id1, track_id2 in collided_track_ids:

                wrong1 = False
                wrong2 = False

                team_assigned_1 = self.match.belongs_to(track_id1)  
                team_assigned_2 = self.match.belongs_to(track_id2)

                distance_1_1 = color_utils.color_distance(colors_per_id[track_id1], self.match.team_1.color)
                distance_1_2 = color_utils.color_distance(colors_per_id[track_id1], self.match.team_2.color)

                
                distance_2_1 = color_utils.color_distance(colors_per_id[track_id2], self.match.team_1.color)
                distance_2_2 = color_utils.color_distance(colors_per_id[track_id2], self.match.team_2.color)

                if distance_1_1 < distance_1_2 and team_assigned_1 == 2: 
                    wrong1 = True

                elif distance_1_1 > distance_1_2 and team_assigned_1 == 1: 
                    wrong1 = True
                
                if distance_2_1 < distance_2_2 and team_assigned_2 == 2: 
                    wrong2 = True

                elif distance_2_1 > distance_2_2 and team_assigned_2 ==1: 
                    wrong2 = True


                if wrong1 and wrong2:
                    # Intercambio total entre jugadores (Caso 1)
                    self.swap_players(track_id1, track_id2)

                elif wrong1 and not wrong2:
                    # track_id1 ha cambiado de equipo incorrectamente (Caso 2)
                    self.reassign_player(track_id1,track_id2)
                
                elif wrong2 and not wrong1:
                    # track_id2 ha cambiado de equipo incorrectamente (Caso 2)
                    self.reassign_player(track_id2, track_id1)
                else: 
                    #print("nothing wrong")
                    pass

    def reassign_player(self, wrong_team_id, wrong_dorsal_id):

        team_1 = self.match.belongs_to(wrong_team_id)  
        team_2 = self.match.belongs_to(wrong_dorsal_id)

        if team_1 != team_2:
            print(f"COLISIÓN CON CAMBIO PARCIAL: NO SON IGUALES {wrong_dorsal_id}, {wrong_team_id}")
            pass

        else:
            if team_1 == 1: 
                wrong_team_key = self.get_key(self.match.team_1.players, wrong_team_id)
                wrong_dorsal_key = self.get_key(self.match.team_1.players, wrong_dorsal_id)

                self.match.team_1.players[wrong_dorsal_key] = None
                self.match.team_1.players[wrong_team_key] = wrong_dorsal_id

                self.match.team_1.get_player_stats_with_id(wrong_dorsal_id).add_track_id(wrong_team_id)

                print("CAMBIO TRAS CHOQUE EN EL EQUIPO 1")
            
            elif team_1 == 2: 

                wrong_team_key = self.get_key(self.match.team_2.players, wrong_team_id)
                wrong_dorsal_key = self.get_key(self.match.team_2.players, wrong_dorsal_id)

                print("cambio de dorsal", wrong_dorsal_id)
                self.match.team_2.players[wrong_dorsal_key] = None
                self.match.team_2.players[wrong_team_key] = wrong_dorsal_id

                self.match.team_2.get_player_stats_with_id(wrong_dorsal_id).add_track_id(wrong_team_id)
                
                #print("CAMBIO TRAS CHOQUE EN EL EQUIPO 2")

            else: 
                #print ("ERROR EN EL CAMBIO DE EQUIPO CON COLISIÓN")
                pass

    def swap_players(self,player_a, player_b):
        
        print(f"Swapping players {player_a} and {player_b}")
        if self.match.team_1.belongs_here(player_a):
            key = self.get_key(self.match.team_1.players, player_a)
            self.match.team_1.players[key] = player_b
            self.match.team_1.get_player_stats_with_id(player_a).add_track_id(player_b)

            key = self.get_key(self.match.team_2.players, player_b)
            self.match.team_2.players[key] = player_a
            self.match.team_2.get_player_stats_with_id(player_b).add_track_id(player_a)

            #print("SWAP DE JUGADORES")
        else:

            key = self.get_key(self.match.team_1.players, player_b)
            self.match.team_1.players[key] = player_a
            print(f"intentando añadir al historial del equipo 1 a {player_a} donde antes estaba {player_b}")
            self.match.team_1.get_player_stats_with_id(player_b).add_track_id(player_a)

            key = self.get_key(self.match.team_2.players, player_a)
            self.match.team_2.players[key] = player_b
            self.match.team_2.get_player_stats_with_id(player_a).add_track_id(player_b)

            #print("SWAP DE JUGADORES")
   
    def detect_n_track(self, frames):
        
        detections = []
        initial = True 
        output_frames = []
        batched_tracks = []
        exists_stub = False
        output_path = "stubs/prueba3.json"

        if os.path.exists(output_path):
            print(f"Archivo {output_path} ya existe. Cargando datos guardados...")
            exists_stub = True
            stub_tracks = stubs_utils.load_batches_from_json(output_path)


        for i in range(0, len(frames), 25):
            
            frame_batch = frames[i:i+25]

            if exists_stub is False:
                #hilo 1
               
                print(f"Procesando batch {(i//25)} / 30 con {len(frame_batch)} frames")
                detections = self.detect_frames(frame_batch)

                #hilo 2
                tracks_by_frame = self.get_tracks(detections) # Devuelve un diccionario de 0 a 24 q contiene otros diccionarios q son player, referee, goalkeeper, ball
                
                batched_tracks.append(tracks_by_frame)
            
            else:
                print(f"Procesando batch {(i//25)} / 30 con {len(frame_batch)} frames")
                tracks_by_frame = stub_tracks[int(i/25)]

            if initial:

                self.assign_referees(tracks_by_frame)
                
                self.assign_teams(color_utils.get_players_colors(frame_batch,tracks_by_frame)) # Obtengo los colores de los jugadores que aparecen en algún frame de los 25 y asigno los equipos 
                
                self.initial_positions(tracks_by_frame)
                initial = False
            
            
            else:
                
                if (new_players := self.check_new_players(tracks_by_frame)):

                    self.assign_new_players(new_players, frame_batch, tracks_by_frame)

                '''if (collided_players := self.check_collision(frame_batch,tracks_by_frame)):
                    
                    self.check_changed_team(collided_players, color_utils.get_players_colors(frame_batch,tracks_by_frame))
                '''
                self.check_wrong_team_assignation(frame_batch,tracks_by_frame)
            #hilo 3 (podemos meter aquí las colisiones también, que realentizan un poco)
            print("EQUIPO 1:", self.match.team_1.players.values())
            print("EQUIPO 2:", self.match.team_2.players.values())
            
            tracks_by_frame = self.ball_interpolator.interpolate_ball(tracks_by_frame)
            
            self.draw_tracks(frame_batch,tracks_by_frame)
            
            self.stats.draw_possession(self.stats.get_posession(tracks_by_frame), tracks_by_frame, frame_batch)
            
            #stats.update_match_possession(batch_number=i)
            
            output_frames = output_frames + frame_batch

        if exists_stub is False:
            stubs_utils.save_batches_to_json(batched_tracks,output_path)

        return output_frames