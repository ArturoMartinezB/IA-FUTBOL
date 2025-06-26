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
        self.H_confidence = 0
        self.stable_keypoints = {}
        self.min_stable_frames = 3
        
        # Dimensiones de las diferentes imágenes de entrada
        self.KEYPOINT_IMG_SIZE = (640, 640)
        self.DETECTION_IMG_SIZE = (1920, 1080)
        self.ORIGINAL_IMG_SIZE = None
        
        # Dimensiones reales del campo (FIFA standard) en METROS
        self.FIELD_WIDTH_M = 120.0  # 120 metros
        self.FIELD_HEIGHT_M = 70.0  # 70 metros
        
        # Dimensiones de la imagen del campo para visualización
        self.FIELD_IMAGE_WIDTH = 1110
        self.FIELD_IMAGE_HEIGHT = 740
        
        # Offsets y área útil del campo en la imagen
        self.field_image_offset_x = 30
        self.field_image_offset_y = 30
        self.FIELD_USABLE_WIDTH = self.FIELD_IMAGE_WIDTH - 2 * self.field_image_offset_x
        self.FIELD_USABLE_HEIGHT = self.FIELD_IMAGE_HEIGHT - 2 * self.field_image_offset_y
        
        current_dir = Path(__file__).parent.absolute()
        project_root = current_dir.parent
        map_route = project_root / "data/field_map.png"
        self.field_image = cv2.imread(str(map_route))
        
        # Coordenadas de los keypoints en METROS (convertidas de centímetros)
        self.cenital_points = np.array([
            [0, 0], [0, 14.5], [0, 25.84], [0, 44.16], [0, 55.5], [0, 70],
            [5.5, 25.84], [5.5, 44.16], [11, 35], [20.15, 14.5],
            [20.15, 25.84], [20.15, 44.16], [20.15, 55.5], [60, 0],
            [60, 25.85], [60, 44.15], [60, 70], [99.85, 14.5],
            [99.85, 25.84], [99.85, 44.16], [99.85, 55.5], [109, 35],
            [114.5, 25.84], [114.5, 44.16], [120, 0], [120, 14.5],
            [120, 25.84], [120, 44.16], [120, 55.5], [120, 70],
            [50.85, 35], [69.15, 35]
        ], dtype=np.float32)

    def redimensionar_frames(self, frames, size=(640, 640)):
        height, width = frames[0].shape[:2]
        
        if self.ORIGINAL_IMG_SIZE is None:
            self.ORIGINAL_IMG_SIZE = (width, height)
        
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
        """Convierte coordenadas de detecciones (1920x1080) a imagen original"""
        if self.ORIGINAL_IMG_SIZE is None:
            return detection_coords
            
        orig_w, orig_h = self.ORIGINAL_IMG_SIZE
        det_w, det_h = self.DETECTION_IMG_SIZE
        
        scale_x = orig_w / det_w
        scale_y = orig_h / det_h
        
        if isinstance(detection_coords, (list, tuple)) and len(detection_coords) == 2:
            return [detection_coords[0] * scale_x, detection_coords[1] * scale_y]
        else:
            converted_coords = []
            for point in detection_coords:
                if len(point) >= 2:
                    orig_x = point[0] * scale_x
                    orig_y = point[1] * scale_y
                    converted_coords.append([orig_x, orig_y])
            return converted_coords

    def update_stable_keypoints(self, detected_keypoints, frame_num):
        """Actualiza el historial de keypoints estables"""
        original_keypoints = self.convert_keypoints_to_original(detected_keypoints)
        
        for i, point in enumerate(original_keypoints):
            if len(point) >= 2 and (point[0] != 0 or point[1] != 0):
                if i not in self.stable_keypoints:
                    self.stable_keypoints[i] = []
                
                self.stable_keypoints[i].append((frame_num, point))
                
                if len(self.stable_keypoints[i]) > self.min_stable_frames * 2:
                    self.stable_keypoints[i] = self.stable_keypoints[i][-self.min_stable_frames * 2:]

    def get_stable_keypoints(self, current_frame):
        """Obtiene keypoints que han sido estables en los últimos frames"""
        stable_cenital = []
        stable_detected = []
        
        for keypoint_idx, history in self.stable_keypoints.items():
            if len(history) >= self.min_stable_frames:
                recent_frames = [frame_num for frame_num, _ in history[-self.min_stable_frames:]]
                if max(recent_frames) >= current_frame - 2:
                    recent_points = [point for _, point in history[-self.min_stable_frames:]]
                    avg_point = np.mean(recent_points, axis=0)
                    
                    # CORRECCIÓN: Usar coordenadas reales en metros, no píxeles
                    stable_cenital.append(self.cenital_points[keypoint_idx])
                    stable_detected.append(avg_point)
        
        return (
            np.array(stable_cenital, dtype=np.float32),
            np.array(stable_detected, dtype=np.float32)
        )

    def should_update_homography(self, new_H, stable_points_count):
        """Decide si actualizar la matriz homográfica"""
        if self.H is None:
            return True
        
        if stable_points_count < 6:
            return False
        
        if new_H is None:
            return False
        
        H_diff = np.linalg.norm(self.H - new_H)
        
        if H_diff > 0.1 and H_diff < 2.0:
            return True
        
        return False

    def keypoints_main_function(self, frame_batch, tracks_by_frame):
        output_images = []
        detections = self.prediction(frame_batch)
        
        for frame_num, det in enumerate(detections):
            key_points = det.xy[0]
            
            self.update_stable_keypoints(key_points, frame_num)
            stable_cenital, stable_detected = self.get_stable_keypoints(frame_num)
            
            if len(stable_detected) >= 6:
                new_H = self.get_homography_matrix(stable_detected, stable_cenital)
                
                if self.should_update_homography(new_H, len(stable_detected)):
                    self.H = new_H
                    print(f"Matriz homográfica actualizada en frame {frame_num} con {len(stable_detected)} puntos")

            if self.H is not None:
                players = tracks_by_frame[frame_num]['players']
                team_1_ids = self.match.team_1.players.values()
                team_2_ids = self.match.team_2.players.values()
                team_1_points = []
                team_2_points = []
                distance_1 = []
                distance_2 = []

                for track_id, bbox in players:
                    detection_point = bbox_utils.get_bottom_center(bbox)
                    original_point = self.convert_detection_to_original(detection_point)
                    
                    # CORRECCIÓN: Transformar a coordenadas reales del campo (metros)
                    transformed_point = self.transform_points(self.H, original_point)
                    if transformed_point is not None:
                        # El punto transformado ya está en metros reales del campo
                        real_point = transformed_point[0][0]
                        
                        # Convertir a coordenadas de visualización para el mapa
                        display_point = self.convert_real_to_display(real_point)
                        
                        if track_id in team_1_ids:
                            team_1_points.append(display_point)
                            distance_1.append((track_id, real_point))  # Usar coordenadas reales
                        elif track_id in team_2_ids:
                            team_2_points.append(display_point)
                            distance_2.append((track_id, real_point))  # Usar coordenadas reales

                # Procesar árbitros y balón
                referees = tracks_by_frame[frame_num]['referees']
                ref_points = self.get_transformed_points_for_display(referees)
                
                ball = tracks_by_frame[frame_num]['ball']
                ball_points = self.get_transformed_points_for_display(ball)

                # Dibujar mapa
                field_image = self.field_image.copy()
                field_image = self.paint_field_map(field_image, team_1_points, self.match.team_1.color)
                field_image = self.paint_field_map(field_image, team_2_points, self.match.team_2.color)
                field_image = self.paint_field_map(field_image, ref_points, (255, 255, 0))
                field_image = self.paint_field_map(field_image, ball_points, (0, 255, 255))

                output_images.append(field_image)

                # CORRECCIÓN: Actualizar distancias usando coordenadas reales
                if frame_num == 0:
                    self.update_player_distance(distance_1, 1)
                    self.update_player_distance(distance_2, 2)
            else:
                output_images.append(self.field_image.copy())

        return output_images

    def convert_real_to_display(self, real_point):
        """Convierte coordenadas reales del campo (metros) a píxeles de visualización"""
        x_m, y_m = real_point
        
        # Escalas: píxeles por metro
        scale_x = self.FIELD_USABLE_WIDTH / self.FIELD_WIDTH_M
        scale_y = self.FIELD_USABLE_HEIGHT / self.FIELD_HEIGHT_M
        
        x_px = x_m * scale_x + self.field_image_offset_x
        y_px = y_m * scale_y + self.field_image_offset_y
        
        return [x_px, y_px]

    def get_transformed_points_for_display(self, tracks):
        """Transforma puntos para visualización"""
        points = []
        if self.H is not None:
            for _, bbox in tracks:
                detection_point = bbox_utils.get_bottom_center(bbox)
                original_point = self.convert_detection_to_original(detection_point)
                
                transformed = self.transform_points(self.H, original_point)
                if transformed is not None:
                    real_point = transformed[0][0]
                    display_point = self.convert_real_to_display(real_point)
                    points.append(display_point)
        return points

    def get_homography_matrix(self, source, target):
        """Calcula la matriz homográfica"""
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
        """Transforma puntos usando homografía"""
        try:
            if H is not None:
                points = np.array(points, dtype=np.float32)
                points = points.reshape(-1, 1, 2)
                transformed_points = cv2.perspectiveTransform(points, H)
                return transformed_points
        except Exception as e:
            print(f"Error transformando puntos: {e}")
        return None

    def convert_to_display_coordinates(self, points):
        """Convierte puntos a coordenadas de visualización"""
        display_points = []
        for point in points:
            if len(point) >= 2:
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
        """CORRECCIÓN: Actualiza distancias usando coordenadas reales del campo"""
        for track_id, real_point in points:  # real_point ya está en metros
            team = getattr(self.match, f"team_{team_number}")
            player_stats = team.get_player_stats_with_id(track_id)
            
            if player_stats is not None:
                last_point = player_stats.last_point
                if last_point is None:
                    player_stats.last_point = real_point
                else:
                    # Calcular distancia euclidiana directamente en metros
                    dx = real_point[0] - last_point[0]
                    dy = real_point[1] - last_point[1]
                    distance_meters = np.sqrt(dx*dx + dy*dy)
                    
                    # Convertir a centímetros si es necesario para compatibilidad
                    distance_cm = distance_meters * 100
                    
                    # Aplicar filtro para evitar distancias irreales
                    # Un jugador no puede moverse más de 50 metros entre frames
                    if distance_meters < 50.0:  # Filtro de distancia máxima
                        player_stats.update_distance(distance_cm)
                        player_stats.last_point = real_point
                    else:
                        print(f"Distancia filtrada para jugador {track_id}: {distance_meters:.2f}m")