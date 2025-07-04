import cv2

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
   
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el vídeo en {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()  

    return frames

def write_video(frames, path):

    print("Gurdando video en : ", path)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 25, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

