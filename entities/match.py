from entities.team import Team

class Match: 

    def __init__(self,team_1, team_2):

        self.referees = []
        self.team_1 = team_1
        self.team_2 = team_2

    
    def set_team_1(self, team):

        self.team_1 = team

    def set_team_2(self, team):

        self.team_2 = team

    def add_referee(self,track_id):

        self.referees.append(int(track_id))

    def belongs_to(self, track_id):

        if track_id in self.team_1.players.values():
            return 1
        
        elif track_id in self.team_2.players.values():
            return 2
        
        else: 
            return None
    
    def get_team_by_int(self, int_id):

        if int_id == 1:
            return self.team_1
        
        elif int_id == 1: 
            return self.team_2
        
        else: 
            return None
        
    def update_possession(self, possession, batch_number):

        #cojo la distribución de posesión actual
        pass