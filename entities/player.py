
class Player:

    def __init__(self, dorsal,track_id, team_id):


        self.dorsal = dorsal
        self.team_id = team_id
        self.track_ids = [track_id]
        self.distance = None
        self.max_vel = None
        self.passes = 0
        self.turn_overs = 0

    def update_distance(self, dis):
        self.distance += dis
        
    def add_pass(self):
        self.passes += 1

    def add_turn_over(self):
        self.turn_overs += 1
    
    def add_track_id(self, track_id):

        self.track_ids.append(track_id)

    def had_this_track_id(self,track_id):

        if track_id in self.track_ids:

            return True
    
    def get_dorsal(self, track_id):

        if track_id in self.track_ids: 

            return self.dorsal

    def print_stats(self):

        print("Dorsal: ", self.dorsal)
        #print("Distancia recorrida: ", self.distance)
        #print("Velocidad máxima: ", self.max_vel)
        print("Pases: ", self.passes)
        print("Pérdidas de balón: ", self.turn_overs)