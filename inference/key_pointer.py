import supervision as sv
import cv2
import numpy as np
from utils import drawing_utils, bbox_utils
from pathlib import Path


class KeyPointer:

    def __init__(self,model, match):

        self.model = model
        self.match = match
        self.H = None
        self.H_points_used = 0

        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent
        map_route = project_root / "data/field_map.png"

        self.field_image = cv2.imread(str(map_route))


        self.field_image_offset_x = 30 # En la foto del campo que utilizo de fondo para el mapeado, hay una distancia entre las líneas del campo y el pixel 0,0 de la imagen,
        self.field_image_offset_y = 30   
        self.cenital_points= np.array([
            [0, 0], [0, 1450.0],[0, 2584.0],[0, 4416.0],[0, 5550.0],[0, 7000],[550, 2584.0],[550, 4416.0],
            [1100, 3500.0],[2015, 1450.0],[2015, 2584.0],[2015, 4416.0],[2015, 5550.0],
            [6000.0, 0],[6000.0, 2585.0],[6000.0, 4415.0],[6000.0, 7000],
            [9985, 1450.0],[9985, 2584.0],[9985, 4416.0],[9985, 5550.0],[10900, 3500.0],[11450, 2584.0],[11450, 4416.0],
            [12000, 0],[12000, 1450.0], [12000, 2584.0],[12000, 4416.0],[12000, 5550.0],[12000, 7000],
            [5085.0, 3500.0],[6915.0, 3500.0]], dtype=np.float32)
        self.edges = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [9, 10], [10, 11],
            [11, 12], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20],
            [22, 23], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29],
            [0, 13], [1, 9], [2, 6], [3, 7], [4, 12], [5, 16], [13, 24],
            [17, 25], [22, 26], [23, 27], [20, 28], [16, 29]
        ]

    ###OBTENER LOS KEYPOINTS
    def redimensionar_frames(self, frames, size=(640, 640)):
        height, width = frames[0].shape[:2]
        print(f"Tamaño del frame: {width}x{height}")
        frames_redimensionados = []
        for frame in frames:
            resized = cv2.resize(frame, size)  # Redimensiona a (width, height)
            frames_redimensionados.append(resized)
        return frames_redimensionados

    def prediction (self, frames):

        results = self.model.predict(self.redimensionar_frames(frames), save=False , imgsz=640)

        key_points = [sv.KeyPoints.from_ultralytics(res) for res in results]

        return key_points

    ###OBTENER MAPA
    def keypoints_main_function(self,frame_batch, tracks_by_frame):

        output_images = []

        # 1 Obtención de los keypoints
        detections = self.prediction(frame_batch)

        keypoints_per_frame = []
        
        for frame_num, det in enumerate(detections):

            #Me quedo solo con las coordenadas de los keypoints
            key_points = det.xy[0]
            keypoints_per_frame.append(key_points)

            #Escojo solo los keypoints detectados para crear la matriz homográfica
            inframe_cenital_points , inframe_detected_keypoints = self.select_inframe_points(key_points)

            #midiendo la distancia euclídea entre las dos matrices homográficas (como si fueran vectores de 9 componentes). Es una forma rápida de saber cuánto han cambiado.
            #Para evitar los cambios continuos en el mapa
            if len(inframe_detected_keypoints) >= 4:
                H_new = self.get_homography_matrix(inframe_detected_keypoints,inframe_cenital_points)
                if self.H is None or np.linalg.norm(H_new - self.H) < 3:
                    self.H = H_new
                    self.H_points_used = len(inframe_detected_keypoints)

                '''Esto lo descarto porque los cambios de matriz holográfica lo único que hacen es mover bruscamente los puntos, no veo mejoras en la representación en el mapa
                elif abs(len(inframe_detected_keypoints) - self.H_points_used) > 4:
                    self.H = H_new
                    self.H_points_used = len(inframe_detected_keypoints)
                '''
            # Una vez tengo la matriz puedo convertir los centros de los bboxes en coordenadas dentro del campo de futbol PARA DIBUJARLOS
            players = tracks_by_frame[frame_num]['players']
            team_1_ids = self.match.team_1.players.values()
            team_2_ids = self.match.team_2.players.values()
            team_1_points = []
            team_2_points = []

            distance_1 = []
            distance_2 = []

            for track_id, bbox in players:

                point = bbox_utils.get_bottom_center(bbox)
                point= self.transform_points(self.H, point)
                point = point[0][0]

                if track_id in team_1_ids:
                    
                    team_1_points.append(point)
                    distance_1.append((track_id, point))

                elif track_id in team_2_ids:
                    team_2_points.append(point)
                    distance_2.append((track_id, point))
            

            referees = tracks_by_frame[frame_num]['referees']
            ref_points =self.get_points(referees)

            ball = tracks_by_frame[frame_num]['ball']
            ball_points = self.get_points(ball)

            field_image= self.field_image.copy()
            field_image = self.paint_field_map(field_image, points=self.resize_points_to1110x740(team_1_points), color=self.match.team_1.color)
            field_image = self.paint_field_map(field_image,inframe_cenital_points,(0,0,0))
            field_image = self.paint_field_map(field_image, self.resize_points_to1110x740(team_2_points), self.match.team_2.color)
            field_image = self.paint_field_map(field_image, self.resize_points_to1110x740(ref_points), (255, 255, 0))
            field_image = self.paint_field_map(field_image, self.resize_points_to1110x740(ball_points), (0, 255, 255))

            output_images.append(field_image)

            #Utilizo los puntos convertidos para obtener las distancias reales recorridas por los jugadores y también las velocidades
            if frame_num == 0:
                self.update_player_distance(distance_1, 1)
                self.update_player_distance(distance_2, 2)
           
        return output_images
    
    def get_points(self, tracks):

        points = []
        for _ , bbox in tracks:
                point = bbox_utils.get_bottom_center(bbox)
                point = self.transform_points(self.H, point)
                point = point[0][0]

                points.append(point)

        return points
    

    
    #OBTENER MATRIZ HOMOGRÁFICA PARA TRANSORMAR LOS BBOX A COORDENADAS EN EL CAMPO
    def select_inframe_points(self, keypoints):

        inframe_cenital_points = []
        inframe_detected_keypoints = []
        #Para hacer la matriz voy a tomar solo los puntos que se hayan obtenido en la detección y su correspondiente homólogo del array de puntos cenitales
        for index, point in enumerate(keypoints):
            if point[0] != 0 or point[1] != 0:

                converted_to_px_cenital_point = self.convert_cm_to_px(self.cenital_points[index])
                inframe_cenital_points.append(converted_to_px_cenital_point)
                inframe_detected_keypoints.append(point)

        return (
        np.array(inframe_cenital_points, dtype=np.float32),
        np.array(inframe_detected_keypoints, dtype=np.float32)
        )


    #Si los cenitales son el source y los keypoints el target estoy plasmando los puntos cenitales sobre el video
    def get_homography_matrix(self, source, target):

        H, mask = cv2.findHomography(source, target, method=cv2.RANSAC)

        return H
    
    def transform_points(self, H, points):

        points = np.array(points, dtype=np.float32)
        points = points.reshape(-1, 1, 2)  # Necesario para cv2.perspectiveTransform
        transformed_points = cv2.perspectiveTransform(points, H)

        return transformed_points

    #Para convertir las coordenadas de los jugadores en posiciones en pixeles para el mapa que tiene un tamaño de 740x1110 px
    def convert_cm_to_px(self, point):

        scale_x = 1050/ 12000  #
        scale_y = 680 / 7000    # 

        x, y = point
        # Convertir de cm reales a píxeles en la imagen cenital
        x_px = x * scale_x + self.field_image_offset_x
        y_px = y * scale_y + self.field_image_offset_y

        return (x_px,y_px)


    def paint_field_map(self, field_image , points, color):

        field = field_image
        for x, y in points:
            #print(f"X = {x} Y= {y}")
            field = drawing_utils.draw_point(field, x, y, color)

        return field

    def resize_points_to1110x740(self, points):

        resized_points =[]

         
        for x, y in points:
            x_real = x * ( 1080/ 1920)
            y_real = y * ( 710 / 1080)
            point = [x_real, y_real]
            resized_points.append(point)

        return resized_points
        
    def update_player_distance(self, points, team_number):

        for track_id, point in points:

            team = getattr(self.match, f"team_{team_number}")
            player_stats = team.get_player_stats_with_id(track_id)
            
            if player_stats is not None:
                last_point = player_stats.last_point
                if player_stats.last_point is None:

                    player_stats.last_point = point
                else:
                    distance = bbox_utils.euclidean_distance_points(last_point,point)
                    player_stats.update_distance(distance)

            

