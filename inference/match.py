from inference.team import Team

class Match: 

    def __init__(self,team_1, team_2):

        self.referees = []
        self.team_1 = team_1
        self.team_2 = team_2
        self.result = (0,0)
        self.possession_percentage = (100,0)
        self.possessor = team_1
    
    def set_team_1(self, team):

        self.team_1 = team

    def set_team_2(self, team):

        self.team_2 = team

    def add_referee(self,track_id):

        self.referees.append(int(track_id))