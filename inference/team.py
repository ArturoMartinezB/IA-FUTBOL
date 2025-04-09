

class Team:

    def __init__(self, name, color):

        self.name = name
        self.color = color 
        self.players = []
        self.goalkeeper = None

    def assign_team_color(self, color):
        self.color = color
    
    def get_team_color(self):
        return self.color
    
    def add_player(self, player):


        #De momento player = track_id
        self.players.append(player)

    def belongs_here(self, player):

        return player in self.players

