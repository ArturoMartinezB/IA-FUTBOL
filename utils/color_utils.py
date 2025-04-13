import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_color_player(image):
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("La imagen no tiene 3 canales (RGB).")
        return None
    
    top_half_image = image[0:int(image.shape[0]/2),:]

    image2D =  top_half_image.reshape(-1,3)

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

    ''' Esto entrena el modelo de clasificación y asigna a los jugadores'''
    kmeans.fit(colors) 

    # Paso 3: Obtener las etiquetas de cada color (a qué clúster pertenece)
    labels = kmeans.labels_  # Mismo orden que `track_ids`
    '''
    # Crear figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Mostrar los puntos con sus colores reales
    ax.scatter(
        colors[:, 0], colors[:, 1], colors[:, 2],
        c=colors / 255,  # Normalizar RGB para matplotlib
        s=100, edgecolors='k'
    )

    # Etiquetas de los ejes
    ax.set_xlabel('Red', fontsize=12)
    ax.set_ylabel('Green', fontsize=12)
    ax.set_zlabel('Blue', fontsize=12)

    # Activar cuadrículas
    ax.grid(True)

    # Rango de los ejes para que abarque todo el espacio RGB
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    ax.set_title('Espacio RGB de los colores de los jugadores', fontsize=14)
    plt.tight_layout()
    plt.show()
    '''

    # Paso 4: Asociar cada track_id a su equipo (clúster)    {k:track_id, v:0} el value es 0 o 1 según sea de un equipo u otro
    track_id_to_team = {track_id: int(label) for track_id, label in zip(track_ids, labels)}
 
    # Paso 5: Obtener el color medio de cada equipo (centroide del clúster)
    team_colors = kmeans.cluster_centers_  # shape: (2, 3)

    return (team_colors,track_id_to_team)