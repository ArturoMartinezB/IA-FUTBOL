import cv2

def draw_ellipse(frame, color, bbox):
    
    x1, _, x2, y2 = bbox
    center = (int((x1 + x2) / 2), y2)

    ellipse = cv2.ellipse(
                frame, 
                center, 
                axes = (int(bbox[2] / 2), int(bbox[3] / 2)),
                angle=0,
                startAngle = -40, 
                endAngle=240, 
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )
    return ellipse


def draw_banner(frame, color, bbox, track_id):
    
    x1, _, x2, y2 = bbox
    center = (int((x1+x2)/2), int(y2))

    rectangle_width = 40
    rectangle_height=20
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
            (0,0,0),
            2
        )

    return frame

