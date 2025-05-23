from sklearn.cluster import KMeans
import numpy as np
class TeamAssigner:
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {} 

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        h, w, _ = frame.shape
        y1 = max(0, int(bbox[1]))
        y2 = min(h, int(bbox[3]))
        x1 = max(0, int(bbox[0]))
        x2 = min(w, int(bbox[2]))
        image = frame[y1:y2, x1:x2]

        if image.size == 0:
            return np.array([0, 0, 0])

        top_half_image = image[0:int(image.shape[0] / 2), :]
        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_cluster = [clustered_image[0,0], clustered_image[0,-1],
                          clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_info in player_detections.items():
            bbox = player_info["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(player_colors)
        self.kmeans = kmeans
        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id
        return team_id
