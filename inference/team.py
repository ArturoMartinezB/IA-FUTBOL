from exceptions.custom_exceptions import TeamFullException

class Team:

    def __init__(self, name):

        self.name = name
        self.color = None 
        self.players = {number: None for number in range(2, 12)}
        self.goalkeeper = None

    def assign_team_color(self, color):
        self.color = color
    
    def add_player(self, track_id):

        for dorsal in range(2,11):
            if self.players[dorsal] == None:
                self.players[dorsal] = track_id
                return True

        #Si llega a este punto, no hay dorsales libres --> El jugador que se intenta añadir ya ha aparecido con anterioridad    
        return False
        

    def belongs_here(self, player):

        return player in self.players.values()

