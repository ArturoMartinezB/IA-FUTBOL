import numpy as np
from sklearn.cluster import KMeans

def get_color_player(image):
    
    top_half_image = image[0:int(image.shape[0]/2),:]

    image2D =  image.reshape(-1,3)

    # Preform K-means with 2 clusters para separar el color del jugador y el fondo
    kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
    kmeans.fit(image2D)

    # Get the cluster labels for each pixel
    labels = kmeans.labels_

    # Reshape the labels to the image shape asignación binaria a cada pixel según el cluster
    clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

    # Se obtiene el cluster de las esquinas que se asume que es el fondo
    corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
    non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)

    #Se obtiene el cluster del jugador a partir del anterior
    player_cluster = 1 - non_player_cluster

    #Se toma el centroide del cluster como el color final
    player_color = kmeans.cluster_centers_[player_cluster]

    return player_color

def get_teams_colors(players_colors):
    
    pass