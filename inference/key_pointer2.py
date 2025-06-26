import supervision as sv
import cv2
import numpy as np
from utils import drawing_utils, bbox_utils
from pathlib import Path


class KeyPointer:

    def __init__(self, model, match):
        self.model = model
        self.match = match
        self.H = None
        self.H_confidence = 0  # Confianza de la matriz actual
        self.stable_keypoints = {}  # Historial de keypoints para estabilidad
        self.min_stable_frames = 3  # Frames mínimos para considerar un keypoint estable
        
        # Dimensiones de las diferentes imágenes de entrada
        self.KEYPOINT_IMG_SIZE = (640, 640)  # Imágenes para detección de keypoints
        self.DETECTION_IMG_SIZE = (1920, 1080)  # Imágenes para detección de jugadores
        self.ORIGINAL_IMG_SIZE = None  # Se establecerá con el primer frame
        
        # Dimensiones reales del campo (FIFA standard)
        self.FIELD_WIDTH_CM = 12000  # 120m
        self.FIELD_HEIGHT_CM = 7000   # 70m
        
        # Dimensiones de la imagen del campo
        self.FIELD_IMAGE_WIDTH = 1110
        self.FIELD_IMAGE_HEIGHT = 740
        
        # Offsets y área útil del campo en la imagen
        self.field_image_offset_x = 30  # Las líneas del campo empiezan a 30px del borde
        self.field_image_offset_y = 30
        self.FIELD_USABLE_WIDTH = self.FIELD_IMAGE_WIDTH - 2 * self.field_image_offset_x  # 1050px
        self.FIELD_USABLE_HEIGHT = self.FIELD_IMAGE_HEIGHT - 2 * self.field_image_offset_y  # 680px
        
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent
        map_route = project_root / "data/field_map.png"
        self.field_image = cv2.imread(str(map_route))

        self.field_image_offset_x = 30
        self.field_image_offset_y = 30
        
        # Coordenadas de los keypoints en centímetros reales
        self.cenital_points = np.array([
            [0, 0], [0, 1450.0],[0, 2584.0],[0, 4416.0],[0, 5550.0],[0, 7000],
            [550, 2584.0],[550, 4416.0], [1100, 3500.0],[2015, 1450.0],
            [2015, 2584.0],[2015, 4416.0],[2015, 5550.0], [6000.0, 0],
            [6000.0, 2585.0],[6000.0, 4415.0],[6000.0, 7000], [9985, 1450.0],
            [9985, 2584.0],[9985, 4416.0],[9985, 5550.0], [10900, 3500.0],
            [11450, 2584.0],[11450, 4416.0], [12000, 0],[12000, 1450.0],
            [12000, 2584.0],[12000, 4416.0],[12000, 5550.0],[12000, 7000],
            [5085.0, 3500.0],[6915.0, 3500.0]
        ], dtype=np.float32)

    def redimensionar_frames(self, frames, size=(640, 640)):
        height, width = frames[0].shape[:2]
        
        # Establecer el tamaño original si no se ha hecho
        if self.ORIGINAL_IMG_SIZE is None:
            self.ORIGINAL_IMG_SIZE = (width, height)
            #print(f"Tamaño original establecido: {width}x{height}")
        
        #print(f"Redimensionando de {width}x{height} a {size[0]}x{size[1]}")
        frames_redimensionados = []
        for frame in frames:
            resized = cv2.resize(frame, size)
            frames_redimensionados.append(resized)
        return frames_redimensionados

    def prediction(self, frames):
        results = self.model.predict(self.redimensionar_frames(frames), save=False)
        key_points = [sv.KeyPoints.from_ultralytics(res) for res in results]
        return key_points

    def convert_keypoints_to_original(self, keypoint_coords):
        """Convierte coordenadas de keypoints (640x640) a imagen original"""
        if self.ORIGINAL_IMG_SIZE is None:
            return keypoint_coords
            
        orig_w, orig_h = self.ORIGINAL_IMG_SIZE
        kp_w, kp_h = self.KEYPOINT_IMG_SIZE
        
        # Calcular escalas de conversión
        scale_x = orig_w / kp_w
        scale_y = orig_h / kp_h
        
        converted_coords = []
        for point in keypoint_coords:
            if len(point) >= 2 and (point[0] != 0 or point[1] != 0):
                orig_x = point[0] * scale_x
                orig_y = point[1] * scale_y
                converted_coords.append([orig_x, orig_y])
            else:
                converted_coords.append(point)
        
        return np.array(converted_coords, dtype=np.float32)

    def convert_detection_to_original(self, detection_coords):
        """Convierte coordenadas de detecciones (544x960) a imagen original"""
        if self.ORIGINAL_IMG_SIZE is None:
            return detection_coords
            
        orig_w, orig_h = self.ORIGINAL_IMG_SIZE
        det_w, det_h = self.DETECTION_IMG_SIZE
        
        # Calcular escalas de conversión
        scale_x = orig_w / det_w
        scale_y = orig_h / det_h
        
        if isinstance(detection_coords, (list, tuple)) and len(detection_coords) == 2:
            # Es un punto individual
            return [detection_coords[0] * scale_x, detection_coords[1] * scale_y]
        else:
            # Es una lista de puntos
            converted_coords = []
            for point in detection_coords:
                if len(point) >= 2:
                    orig_x = point[0] * scale_x
                    orig_y = point[1] * scale_y
                    converted_coords.append([orig_x, orig_y])
            return converted_coords

    def update_stable_keypoints(self, detected_keypoints, frame_num):
        """Actualiza el historial de keypoints estables"""
        # Convertir keypoints a espacio de imagen original
        original_keypoints = self.convert_keypoints_to_original(detected_keypoints)
        
        for i, point in enumerate(original_keypoints):
            if len(point) >= 2 and (point[0] != 0 or point[1] != 0):  # Keypoint detectado
                if i not in self.stable_keypoints:
                    self.stable_keypoints[i] = []
                
                self.stable_keypoints[i].append((frame_num, point))
                
                # Mantener solo los últimos N frames
                if len(self.stable_keypoints[i]) > self.min_stable_frames * 2:
                    self.stable_keypoints[i] = self.stable_keypoints[i][-self.min_stable_frames * 2:]

    def get_stable_keypoints(self, current_frame):
        """Obtiene keypoints que han sido estables en los últimos frames"""
        stable_cenital = []
        stable_detected = []
        
        for keypoint_idx, history in self.stable_keypoints.items():
            if len(history) >= self.min_stable_frames:
                # Verificar que el keypoint apareció en frames recientes
                recent_frames = [frame_num for frame_num, _ in history[-self.min_stable_frames:]]
                if max(recent_frames) >= current_frame - 2:  # Tolerancia de 2 frames
                    # Usar el promedio de las últimas detecciones para más estabilidad
                    recent_points = [point for _, point in history[-self.min_stable_frames:]]
                    avg_point = np.mean(recent_points, axis=0)
                    
                    converted_cenital = self.convert_cm_to_field_px(self.cenital_points[keypoint_idx])
                    stable_cenital.append(converted_cenital)
                    stable_detected.append(avg_point)
        
        return (
            np.array(stable_cenital, dtype=np.float32),
            np.array(stable_detected, dtype=np.float32)
        )

    def should_update_homography(self, new_H, stable_points_count):
        """Decide si actualizar la matriz homográfica basándose en estabilidad"""
        if self.H is None:
            return True
        
        if stable_points_count < 6:  # Necesitamos suficientes puntos estables
            return False
        
        if new_H is None:
            return False
        
        # Calcular diferencia entre matrices
        H_diff = np.linalg.norm(self.H - new_H)
        
        # Solo actualizar si la diferencia es significativa pero no excesiva
        if H_diff > 0.1 and H_diff < 2.0:  # Valores ajustables
            return True
        
        return False

    def keypoints_main_function(self, frame_batch, tracks_by_frame):
        output_images = []
        detections = self.prediction(frame_batch)
        
        for frame_num, det in enumerate(detections):
            # Obtener keypoints detectados
            key_points = det.xy[0]
            #print( "KEYPOINTS ==============" , key_points)
            #for track_id, bbox in tracks_by_frame[frame_num]['players']:
            #    print(f"Player {track_id} : Bbox = {bbox}")
            # Actualizar historial de keypoints estables
            self.update_stable_keypoints(key_points, frame_num)
            
            # Obtener keypoints estables para la homografía
            stable_cenital, stable_detected = self.get_stable_keypoints(frame_num)
            
            # Actualizar matriz homográfica solo si es necesario
            if len(stable_detected) >= 6:  # Mínimo 6 puntos para mayor estabilidad
                new_H = self.get_homography_matrix(stable_detected, stable_cenital)
                
                if self.should_update_homography(new_H, len(stable_detected)):
                    self.H = new_H
                    print(f"Matriz homográfica actualizada en frame {frame_num} con {len(stable_detected)} puntos")

            # Procesar jugadores y otros elementos
            if self.H is not None:
                players = tracks_by_frame[frame_num]['players']
                team_1_ids = self.match.team_1.players.values()
                team_2_ids = self.match.team_2.players.values()
                team_1_points = []
                team_2_points = []
                distance_1 = []
                distance_2 = []

                for track_id, bbox in players:
                    # Obtener punto del bbox (en espacio 544x960)
                    detection_point = bbox_utils.get_bottom_center(bbox)
                    
                    # Convertir a espacio de imagen original
                    original_point = self.convert_detection_to_original(detection_point)
                    
                    # Transformar usando la matriz homográfica
                    transformed_point = self.transform_points(self.H, original_point)
                    if transformed_point is not None:
                        point_2d = transformed_point[0][0]
                        
                        if track_id in team_1_ids:
                            team_1_points.append(point_2d)
                            distance_1.append((track_id, point_2d))
                        elif track_id in team_2_ids:
                            team_2_points.append(point_2d)
                            distance_2.append((track_id, point_2d))

                # Procesar árbitros y balón
                referees = tracks_by_frame[frame_num]['referees']
                ref_points = self.get_transformed_points(referees)
                
                ball = tracks_by_frame[frame_num]['ball']  
                ball_points = self.get_transformed_points(ball)

                # Dibujar mapa
                field_image = self.field_image.copy()
                field_image = self.paint_field_map(field_image, 
                                                 self.convert_to_display_coordinates(team_1_points), 
                                                 self.match.team_1.color)
                field_image = self.paint_field_map(field_image, 
                                                 self.convert_to_display_coordinates(stable_cenital), 
                                                 (0,0,0))
                field_image = self.paint_field_map(field_image, 
                                                 self.convert_to_display_coordinates(team_2_points), 
                                                 self.match.team_2.color)
                field_image = self.paint_field_map(field_image, 
                                                 self.convert_to_display_coordinates(ref_points), 
                                                 (255, 255, 0))
                field_image = self.paint_field_map(field_image, 
                                                 self.convert_to_display_coordinates(ball_points), 
                                                 (0, 255, 255))

                output_images.append(field_image)

                # Actualizar distancias de jugadores
                if frame_num == 0:
                    self.update_player_distance(distance_1, 1)
                    self.update_player_distance(distance_2, 2)
            else:
                # Si no hay matriz homográfica, usar imagen de campo vacía
                output_images.append(self.field_image.copy())

        return output_images

    def get_transformed_points(self, tracks):
        """Transforma puntos usando la matriz homográfica actual"""
        points = []
        if self.H is not None:
            for _, bbox in tracks:
                # Obtener punto del bbox (en espacio 544x960)
                detection_point = bbox_utils.get_bottom_center(bbox)
                
                # Convertir a espacio de imagen original
                original_point = self.convert_detection_to_original(detection_point)
                
                # Transformar usando la matriz homográfica
                transformed = self.transform_points(self.H, original_point)
                if transformed is not None:
                    points.append(transformed[0][0])
        return points

    def get_homography_matrix(self, source, target):
        """Calcula la matriz homográfica con manejo de errores"""
        try:
            if len(source) >= 4 and len(target) >= 4:
                H, mask = cv2.findHomography(source, target, 
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=5.0,
                                           maxIters=2000,
                                           confidence=0.995)
                return H
        except Exception as e:
            print(f"Error calculando homografía: {e}")
        return None

    def transform_points(self, H, points):
        """Transforma puntos con manejo de errores"""
        try:
            if H is not None:
                points = np.array(points, dtype=np.float32)
                points = points.reshape(-1, 1, 2)
                transformed_points = cv2.perspectiveTransform(points, H)
                return transformed_points
        except Exception as e:
            print(f"Error transformando puntos: {e}")
        return None

    def convert_cm_to_field_px(self, point_cm):
        """Convierte coordenadas de cm reales a píxeles en la imagen del campo"""
        # Escalas: área útil del campo en píxeles / dimensiones reales en cm
        scale_x = self.FIELD_USABLE_WIDTH / self.FIELD_WIDTH_CM  # 1050px / 12000cm = 0.0875
        scale_y = self.FIELD_USABLE_HEIGHT / self.FIELD_HEIGHT_CM  # 680px / 7000cm = 0.0971
        
        x, y = point_cm
        # Convertir de cm a píxeles dentro del área útil, luego añadir offset
        x_px = x * scale_x + self.field_image_offset_x
        y_px = y * scale_y + self.field_image_offset_y
        
        return (x_px, y_px)

    def convert_to_display_coordinates(self, points):
        """Convierte puntos transformados a coordenadas de visualización"""
        display_points = []
        for point in points:
            if len(point) >= 2:
                # Las coordenadas ya están en el espacio del campo de visualización
                # Solo necesitamos asegurar que estén en el rango correcto
                #print(f"Punto: x = {point[0]} , y ={point[1]}")
                x = max(0, min(point[0], self.FIELD_IMAGE_WIDTH))
                y = max(0, min(point[1], self.FIELD_IMAGE_HEIGHT))
                display_points.append([x, y])
        return display_points

    def paint_field_map(self, field_image, points, color):
        """Dibuja puntos en el mapa del campo"""
        field = field_image.copy()
        for point in points:
            if len(point) >= 2:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < field_image.shape[1] and 0 <= y < field_image.shape[0]:
                    field = drawing_utils.draw_point(field, x, y, color)
        return field

    def update_player_distance(self, points, team_number):
        """Actualiza las distancias recorridas por los jugadores"""
        for track_id, point in points:
            team = getattr(self.match, f"team_{team_number}")
            player_stats = team.get_player_stats_with_id(track_id)
            
            if player_stats is not None:
                last_point = player_stats.last_point
                if last_point is None:
                    player_stats.last_point = point
                else:
                    # Calcular distancia en píxeles
                    distance_px = bbox_utils.euclidean_distance_points(last_point, point)
                    
                    # Convertir píxeles a centímetros reales
                    # Usar escalas inversas: cm por píxel
                    scale_x_cm_per_px = self.FIELD_WIDTH_CM / self.FIELD_USABLE_WIDTH  # 12000cm / 1050px
                    scale_y_cm_per_px = self.FIELD_HEIGHT_CM / self.FIELD_USABLE_HEIGHT  # 7000cm / 680px
                    
                    # Usar promedio de escalas para distancia euclidiana
                    avg_scale = (scale_x_cm_per_px + scale_y_cm_per_px) / 2
                    distance_cm = distance_px * avg_scale
                    
                    player_stats.update_distance(distance_cm)
                    player_stats.last_point = point
