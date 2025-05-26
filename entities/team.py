from entities.player import Player

class Team:

    def __init__(self, id):

        self.team_id = id
        self.color = None 
        self.players = {number: None for number in range(2, 12)}
        self.players_stats = {number: None for number in range(2, 12)}
        self.goalkeeper = None
        self.last_position = {}

    def assign_team_color(self, color):
        self.color = color
    
    def add_player(self, track_id):

        if track_id not in list(self.players.values()):
            for dorsal in range(2,12):
                if self.players[dorsal] == None:
                    self.players[dorsal] = int(track_id)
                    self.players_stats[dorsal] = Player(dorsal, int(track_id), int(self.team_id))
                    print(f"jugador {track_id} añadido")
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

    def get_player_stats(self, dorsal):

        return self.players_stats[dorsal]
    
    def get_player_stats_with_id(self, track_id):

        #print("Track_id buscado: " , track_id)
        for dorsal in range(2,12):
            
            if self.players_stats[dorsal] is not None:
                
                #print(f"Track_ids del jugador del equipo {self.team_id} son : {self.players_stats[dorsal].track_ids}")
                if track_id in self.players_stats[dorsal].track_ids:
                    
                    return self.players_stats[dorsal]
            
        print("NO SE HAN PODIDO OBTENER LAS STATS DEL JUGADOR CON ESE TRACK_ID")
        return None

    
    def add_pass(self,track_id):

        dorsal = self.get_dorsal(track_id)

        player = self.players_stats[dorsal]

        player.add_pass()

    def add_turn_over(self, track_id):

        dorsal = self.get_dorsal(track_id)

        player = self.players_stats[dorsal]

        player.add_turn_over()

    def print_players_stats(self):

        for i in range(2,12): 
            player_stats = self.players_stats[i]

            if player_stats is not None: 

                player_stats.print_stats()