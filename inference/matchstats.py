import supervision as sv
import numpy as np
from utils import bbox_utils

class MatchStats:

    def __init__(self, match):

        self.match = match
        self.last_possessor = None

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

                
    def get_posession(self, tracks_by_frame):

        for num, _ in enumerate(tracks_by_frame):
            
            if tracks_by_frame[num].get('ball'):

                #Quiero calcular para cada frame, el jugador más cercano al balón, que este a menos de 75 unidades de distancia
                
                ball = tracks_by_frame[num].get('ball')[0]

                ball_bbox = ball[1]
                
                possessor = self.get_possessor(ball_bbox, tracks_by_frame[num]['players'])

                
                if possessor != -1 and self.last_possessor is None:

                    self.last_possessor = possessor
                
                elif possessor != -1 and possessor != self.last_possessor:
                    
                    self.change_in_possession(possessor, self.last_possessor)


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
        
    def check_change_of_possession(self, player1_id, player2_id):
        
        if player1_id is None or player2_id is None:

            print("check_change_of_possession ha recibido un None como jugador")
            return None

        if self.match.belongs_to(player1_id) == self.match.belongs_to(player2_id):

            # HA REALIZADO UN PASE
            self.update_pass(player1_id)
            self.last_possessor = player2_id

            if 1 == self.match.belongs_to(player1_id):
                print(f"El jugador {self.match.team_1.get_dorsal(player1_id)}, se la ha pasado a {self.match.team_1.get_dorsal(player2_id)}")
            
            else:
                print(f"El jugador {self.match.team_2.get_dorsal(player1_id)} ha perdido el balón y lo ha interceptado {self.match.team_2.get_dorsal(player2_id)}")

        elif self.match.belongs_to(player1_id) != self.match.belongs_to(player2_id):
            
            self.update_turn_over(player1_id)
            self.last_possessor = player2_id

            if 1 == self.match.belongs_to(player1_id):
                print(f"El jugador {self.match.team_1.get_dorsal(player1_id)} ha perdido el balón y lo ha interceptado {self.match.team_2.get_dorsal(player2_id)}")
            
            else:
                print(f"El jugador {self.match.team_2.get_dorsal(player1_id)} ha perdido el balón y lo ha interceptado {self.match.team_1.get_dorsal(player2_id)}")

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
    

    def update_match_possession(self,batch_number):
        pass