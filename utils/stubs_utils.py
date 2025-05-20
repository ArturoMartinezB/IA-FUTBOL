import json
import os
import numpy as np

def save_batches_to_json(batches_tracks, output_path):
        serializable_data = []

        for batch_num, tracks_by_frame in enumerate(batches_tracks):
            batch_data = {
                "batch": batch_num,
                "tracks_by_frame": {
                    str(int(f)): {
                        key: [[int(tid) if isinstance(tid, (np.integer, np.int64)) else tid,
                            [float(coord) for coord in bbox]]
                            for tid, bbox in value]
                        for key, value in frame_data.items()
                    } for f, frame_data in tracks_by_frame.items()
                }
            }
            serializable_data.append(batch_data)

        with open(output_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

def load_batches_from_json(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    batches_tracks = []
    cont = 0
    for batch in data:
        tracks_by_frame = {
            int(f): {
                key: [(tid, bbox) for tid, bbox in value]
                for key, value in frame_data.items()
            } for f, frame_data in batch["tracks_by_frame"].items()
        }
        batches_tracks.append(tracks_by_frame)
        cont +=1

    return batches_tracks