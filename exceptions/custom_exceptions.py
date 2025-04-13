
class TeamFullException(Exception):
    
    def __init__(self, message="No se pueden añadir más de 10 jugadores de campo al equipo."):
        self.message = message
        super().__init__(self.message)