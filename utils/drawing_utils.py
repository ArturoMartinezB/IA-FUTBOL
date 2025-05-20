import cv2
import numpy as np

def draw_ellipse(frame, color, bbox):
    
    x1, y1, x2, y2 = bbox
    center = (int((x1 + x2) / 2), int(y2)) 

    axes = (int( x2-x1), int((y2-y1)/4))

    cv2.ellipse(
                frame, 
                center, 
                axes,
                angle=0,
                startAngle = -40, 
                endAngle=240, 
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )
    
    return frame

def draw_banner(frame, color, bbox, track_id):
    
    x1, _, x2, y2 = bbox
    center = (int((x1+x2)/2), int(y2))

    rectangle_width = 40
    rectangle_height= 20
    x1_rect = center[0] - rectangle_width//2
    x2_rect = center[0] + rectangle_width//2
    y1_rect = (y2- rectangle_height//2) +15
    y2_rect = (y2+ rectangle_height//2) +15

    if track_id is not None:
        cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect) ),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED)
        
        x1_text = x1_rect+12
        if track_id > 99:
            x1_text -=10
        
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text),int(y1_rect+15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

    return frame

def draw_pointer(frame, bbox,color = (0, 255, 255), size = 15):

    x1, y1, x2, y2 = bbox 

    # Define los tres vértices del triángulo (apunta hacia arriba)
    pt1 = (((x1 + x2) / 2), y1-5)
    pt2 = (((x1 + x2) / 2 + (x2-x1)/3), (y1 - 15))
    pt3 = (((x1 + x2) / 2 - (x2-x1)/3), (y1 - 15))

    triangle_cnt = np.array([pt1, pt2, pt3], dtype=np.int32)

    # Dibuja el contorno del triángulo
    cv2.drawContours(frame, [triangle_cnt], 0, color, 2)
    
    return frame