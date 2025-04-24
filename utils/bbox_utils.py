import math 

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def euclidean_distance(bbox1, bbox2):
    x1, y1 = get_center(bbox1)
    x2, y2 = get_center(bbox2)
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def nearest_bbox(bbox, players):

    nearest_track_id = None
    distance = 100000

    for track_id, bbox_it in players:
        
        distance_it = euclidean_distance(bbox, bbox_it)
        if distance > distance_it:
            distance = distance_it
            nearest_track_id = track_id
    
    return (nearest_track_id)