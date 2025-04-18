"""

X = [
    (0.4, 2.5),
    ...
]


TODO: could partitiion space for faster searching?
"""
from typing import List, Tuple


    
def k_means(X, k: int, max_movement_per_iter: float) -> List[Tuple(float)]: # outer list is len k
    centroids = [
        [5, 8],
        # k random points
    ]
    point_position_sums = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    # centroid_id_to_count = [0] * k
    # while True:
    #     for point in X:
    #         min_dist = float('inf')
    #         min_cent_idx = None
    #         for centroid_idx, centroid in enumerate(centroids):
    #             if curr_dist := dist(point, centroid) < min_dist:
    #                 min_dist = curr_dist
    #                 min_cent_idx = centroid_idx
    #         point_position_sums[min_cent_idx] += point
    #         centroid_id_to_count[centroid_idx] += 1
    #     new_centroids = point_position_sums / centroid_id_to_count
    #     if max(abs(centroids - new_centroids).sum(axis=1)) < max_movement_per_iter:
    #         centroids = new_centroids
    #         break
    #     centroids = new_centroids
    # return centroids

        
        
        
                    
            
            
        