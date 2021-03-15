import numpy as np
from cv2 import cv2
from dominant_colour import get_dominant_colour, visualise_colours

img_url = 'https://images-na.ssl-images-amazon.com/images/I/81HmoJNlUnL._AC_SL1500_.jpg'
dominant, cluster, centroid = get_dominant_colour(img_url, timing=True)

# Display most dominant colors
visualise = visualise_colours(cluster, cluster.cluster_centers_)
visualise = cv2.cvtColor(visualise, cv2.COLOR_RGB2BGR)
cv2.imshow('visualise', visualise)
cv2.waitKey()
