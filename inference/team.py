from exceptions.custom_exceptions import TeamFullException

class Team:

    def __init__(self, name):

        self.name = name
        self.color = None 
        self.players = {number: None for number in range(2, 12)}
        self.goalkeeper = None
        self.last_position = {}

    def assign_team_color(self, color):
        self.color = color
    
    def add_player(self, track_id):

        for dorsal in range(2,12):
            if self.players[dorsal] == None:
                self.players[dorsal] = int(track_id)
                return True

        #Si llega a este punto, no hay dorsales libres --> El jugador que se intenta añadir ya ha aparecido con anterioridad en el video
        return False
    
    def update_last_position(self, track_id, bbox):
        
        self.last_position[int(track_id)] = bbox


    def get_dorsal(self,track_id):

        for it in range(2,12):
            if self.players[it] == int(track_id):
                return it

    def belongs_here(self, player):

        return player in self.players.values()

