import pandas as pd


class BallInterpolator:

    def __init__(self,match_stats):
        
        self.stats = match_stats 

    def interpolate_ball(self, tracks_by_frame):  

        ball_detections = [] 
        interpolable = False

        for num, _ in enumerate(tracks_by_frame):

            if not tracks_by_frame[num].get('ball'):
                
                ball_detections.append([])
                #print("NO HAY BALÓN en frame: ", num)
                #print(tracks_by_frame[num].get('ball'))
            else: 

                interpolable = True
                ball = tracks_by_frame[num]['ball'][0]
                
                track_id = ball[0]
                ball_bbox = ball[1]
            
                #print("BBOX del balón", ball_bbox)
                ball_detections.append(ball_bbox)

        if interpolable:

            df_detections = pd.DataFrame(ball_detections,columns=['x1','y1','x2','y2'])

            df_detections = df_detections.interpolate()
            df_detections = df_detections.bfill()

            for num, _ in enumerate(tracks_by_frame):
                
                if not tracks_by_frame[num].get('ball'):
                    
                    tracks_by_frame[num]['ball'] = [(track_id, df_detections.iloc[num].to_numpy().tolist())]
                    self.stats.ball_interpolations += 1

        return tracks_by_frame