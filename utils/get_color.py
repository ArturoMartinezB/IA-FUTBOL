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
    
    # Paso 1: Obtener los colores como lista de arrays
    track_ids = list(players_colors.keys())
    colors = np.array(list(players_colors.values()))

    # Paso 2: Aplicar KMeans con 2 clusters
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
    kmeans.fit(colors)

    # Paso 3: Obtener las etiquetas de cada color (a qué clúster pertenece)
    labels = kmeans.labels_  # Mismo orden que `track_ids`

    # Paso 4: Asociar cada track_id a su equipo (clúster)
    track_id_to_team = {track_id: int(label) for track_id, label in zip(track_ids, labels)}

    # Paso 5: Obtener el color medio de cada equipo (centroide del clúster)
    team_colors = kmeans.cluster_centers_  # shape: (2, 3)

    return (team_colors,track_id_to_team)