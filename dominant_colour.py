import numpy as np
import pandas as pd
import time
from cv2 import cv2
from sklearn.cluster import KMeans
from skimage import io


def get_dominant_colour(img_url, timing=False):
    if timing:
        start = time.perf_counter()
        tic = time.perf_counter()
    img = io.imread(img_url)
    if timing:
        toc = time.perf_counter()
        print(f"Loaded the image in {toc - tic:0.4f} seconds")
    img = img.reshape((-1, 3))

    if timing:
        tic = time.perf_counter()
    cluster = KMeans(n_clusters=5, n_init=3, max_iter=10, tol=0.001)
    cluster.fit(img)
    if timing:
        toc = time.perf_counter()
        print(f"KMeans calculation in {toc - tic:0.4f} seconds")

    labels = cluster.labels_
    centroid = cluster.cluster_centers_

    if timing:
        tic = time.perf_counter()
    percent = []
    _, counts = np.unique(labels, return_counts=True)
    for i in range(len(centroid)):
        j = counts[i]
        j = j/(len(labels))
        percent.append(j)
    if timing:
        toc = time.perf_counter()
        print(f"Percentage calculation in {toc - tic:0.4f} seconds")

    indices = np.argsort(percent)[::-1]
    dominant = centroid[indices[0]]
    if timing:
        end = time.perf_counter()
        print(f"get_dominant_colour execution in {end - start:0.4f} seconds")

    return dominant, cluster, centroid


def clamp(x):
    return int(max(0, min(x, 255)))


def get_rgb_colour(img_url, debug=False):
    dominant_colour = get_dominant_colour(img_url)
    r = dominant_colour[0]
    g = dominant_colour[1]
    b = dominant_colour[2]

    if debug:
        hex_str = "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))
        print(f'{hex_str}')

    rgb_colour = (clamp(r), clamp(g), clamp(b))

    return rgb_colour


def visualise_colours(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color)
                     for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50),
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect
