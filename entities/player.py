
class Player:

    def __init__(self, dorsal,track_id, team_id):


        self.dorsal = dorsal
        self.team_id = team_id
        self.track_ids = [track_id]
        self.distance = 0.0
        self.max_vel = 0.0
        self.passes = 0
        self.turn_overs = 0
        self.last_point = None
        self.velocity = 0.0

    def update_distance(self, dis):
        self.distance += (dis / 100)
        self.velocity = (dis / 100) # ya que comparo la distancia recorrida en cada frame 0, hasta el siguiente batch falta un segundo con lo que la velocidad es en cm/s 
        
        if (dis / 100) > self.max_vel: 

            self.max_vel = (dis / 100)
        
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

        print("\n")
        print("Dorsal: ", self.dorsal)
        print("------------------------")
        print(f"Distancia recorrida: {self.distance:.2f} m")
        print(f"Velocidad máxima: {self.max_vel:.2f} m/s")
        print("Pases: ", self.passes)
        print("Pérdidas de balón: ", self.turn_overs)

    def get_stats_sheet(self):

        stats_sheet = {
            'dorsal': self.dorsal,
            'distance': self.distance,
            'max_velocity': self.max_vel,
            'passes': self.passes,
            'turnovers': self.turn_overs
        }

        return stats_sheet