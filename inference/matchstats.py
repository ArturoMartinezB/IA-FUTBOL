import supervision as sv
import numpy as np
from utils import bbox_utils , drawing_utils

class MatchStats:

    def __init__(self, match):

        self.match = match
        self.last_possessor = None
        self.possession_1_frames = 0
        self.possession_2_frames = 0

    def get_possessor(self, ball_bbox, players):

        ball_center_point = bbox_utils.get_center(ball_bbox)

        min_distance = 30

        possessor = -1

        for track_id, bbox in players:

            x1, _, x2, y2 = bbox
            left_foot = (x1,y2)
            right_foot  = (x2,y2)

            #Escojo el pie (esquina inferior izquierda o derecha del bbox) que esté más cerca
            left_foot_dis = bbox_utils.euclidean_distance_points(ball_center_point,left_foot)
            right_foot_dis = bbox_utils.euclidean_distance_points(ball_center_point,right_foot)

            distance = min(left_foot_dis, right_foot_dis)

            if distance < min_distance:

                min_distance = distance
                possessor = track_id

        return possessor

           
    def get_match_stats(self, tracks_by_frame):

        possessor_per_frame = []
        for num, _ in enumerate(tracks_by_frame):
            
            if tracks_by_frame[num].get('ball'):

                #Quiero calcular para cada frame, el jugador más cercano al balón, que este a menos de 75 unidades de distancia
                
                ball = tracks_by_frame[num].get('ball')[0]

                ball_bbox = ball[1]
                
                possessor = self.get_possessor(ball_bbox, tracks_by_frame[num]['players'])

                possessor_per_frame.append(possessor)
                
                #Actualizar posesión del partido
                self.update_match_possession(possessor)

                if possessor != -1 and self.last_possessor is None:

                    self.last_possessor = possessor
                
                elif possessor != -1 and possessor != self.last_possessor:
                    
                    self.change_in_possession(possessor, self.last_possessor)

            else:
                possessor_per_frame.append(None)

            #print(f"Last Possessor frame: {num} = {self.last_possessor}")

        return possessor_per_frame  #devuelve un array con los poseedores en cada frame, o None si no hay nadie con el balón

    #ESTA FUNCIÓN ES IMPORTANTE, PORQUE COMO ESTOY TRABAJANDO CON TRACK_IDS DEL TRACKS_BY_FRAME, Y COMO EL BATCH YA ESTÁ PROCESADO A NIVEL DE ASIGNACIÓN DE DORSAL = TRACK_ID
    #PUEDE QUE UN TRACK_ID QUE APARECE EN LOS PRIMEROS FRAMES, LUEGO DESAPAREZCA Y NO TENGA UN DORSAL ASIGNADO, CON LO QUE NO SE PODÍA ACCEDER A ESA PLAYER_SHEET
    #HE MODIFICADO PLAYER PARA QUE CONTENGA LOS TRACK_ID QUE HA TENIDO, OBTENER CON ESTE REGISTRO EL DORSAL Y CON EL DORSAL EL NUEVO TRACK_ID.
    def check_change_in_team_list(self, track_id):

        team_id = self.match.belongs_to(track_id)

        if team_id is None:

            if self.match.team_1.get_player_stats_with_id(track_id) is not None: 

                dorsal = self.match.team_1.get_player_stats_with_id(track_id).dorsal

                print("El jugador esta en el equipo uno y su nuevo track_id es:", self.match.team_1.players[dorsal])
                return self.match.team_1.players[dorsal]
            
            elif self.match.team_2.get_player_stats_with_id(track_id) is not None:

                dorsal = self.match.team_2.get_player_stats_with_id(track_id).dorsal

                print("El jugador esta en el equipo dos y su nuevo track_id es:", self.match.team_2.players[dorsal])
                return self.match.team_2.players[dorsal]
            else:
                print("TRACK_ID EN NINGUNA LISTA DE PLAYER_SHEETS")
                return None

        else:  

            #print("El track_id es el mismo:", track_id)
            return track_id
        

    def change_in_possession(self, possessor, last_possessor):

        possessor = self.check_change_in_team_list(possessor)
        last_possessor = self.check_change_in_team_list(last_possessor)

        self.check_change_of_possession(last_possessor, possessor)
        
    def check_change_of_possession(self, last_possessor, possessor):
        
        if last_possessor is None or possessor is None:

            print("check_change_of_possession ha recibido un None como jugador")
            return None

        if self.match.belongs_to(last_possessor) == self.match.belongs_to(possessor):

            # HA REALIZADO UN PASE
            self.update_pass(last_possessor)
            self.last_possessor = possessor

            if 1 == self.match.belongs_to(last_possessor):
                print(f"El jugador {self.match.team_1.get_dorsal(last_possessor)}, se la ha pasado a {self.match.team_1.get_dorsal(possessor)}")
            
            else:
                print(f"El jugador {self.match.team_2.get_dorsal(last_possessor)}, se la ha pasado a {self.match.team_2.get_dorsal(possessor)}")

        elif self.match.belongs_to(last_possessor) != self.match.belongs_to(possessor):
            
            print("el jugador uno pertenece al equipo: ", self.match.belongs_to(last_possessor))
            print("el jugador dos pertenece al equipo: ", self.match.belongs_to(possessor))
            self.update_turn_over(last_possessor)
            self.last_possessor = possessor

            if 1 == self.match.belongs_to(last_possessor):
                print(f"El jugador {self.match.team_1.get_dorsal(last_possessor)} ha perdido el balón y lo ha interceptado {self.match.team_2.get_dorsal(possessor)}")
            
            else:
                print(f"El jugador {self.match.team_2.get_dorsal(last_possessor)} ha perdido el balón y lo ha interceptado {self.match.team_1.get_dorsal(possessor)}")

        else:
            print("check_change_of_possession no entra en los casos,  almenos uno de los players no está en ningún equipo")

    def update_pass(self, player_id):

        team = self.match.belongs_to(player_id)

        if team == 1:

            self.match.team_1.add_pass(player_id)
        else: 

            self.match.team_2.add_pass(player_id)

    def update_turn_over(self, player_id):

        team = self.match.belongs_to(player_id)

        if team == 1:

            self.match.team_1.add_turn_over(player_id)
        else: 

            self.match.team_2.add_turn_over(player_id)
    

    def update_match_possession(self,possessor):

        
        if self.match.team_1.belongs_here(possessor) is True:

            self.possession_1_frames += 1
        
        else:

            self.possession_2_frames += 1

    def get_match_possession(self):

        total_frames = self.possession_1_frames + self.possession_2_frames

        return (float(self.possession_1_frames/total_frames), float(self.possession_2_frames/total_frames))

    def draw_possession(self, possessor_per_frame, tracks_by_frame, frames):

        for num , frame in enumerate(frames):

            if possessor_per_frame[num] is not None:

                possessor = possessor_per_frame[num]

                for track_id, bbox  in tracks_by_frame[num]['players']:

                    if track_id == possessor:

                        drawing_utils.draw_pointer(frame,bbox,(255, 0, 0), 15)

    
    def print_match_stats(self):

        print("Possession: ",self.get_match_possession)