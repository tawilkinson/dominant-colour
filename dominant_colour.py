import numpy as np
import time
from cv2 import cv2
from sklearn.cluster import KMeans
from skimage import io
from skimage.transform import rescale


def cv2_dominant_colour(img_url,  colours=10, timing=False):
    '''
    Dominant Colour method using open cv, based on
    https://stackoverflow.com/a/43111221/2523885
    '''

    if timing:
        start = time.perf_counter()
        tic = time.perf_counter()
    img = io.imread(img_url)
    pixels = np.float32(img.reshape(-1, 3))
    if timing:
        toc = time.perf_counter()
        print(f"Loaded the image in {toc - tic:0.2f}s")

    if timing:
        tic = time.perf_counter()
    n_colours = colours
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centroid = cv2.kmeans(
        pixels, n_colours, None, criteria, 10, flags)
    labels = labels.flatten().tolist()

    _, counts = np.unique(labels, return_counts=True)
    if timing:
        toc = time.perf_counter()
        print(f"KMeans calculation in {toc - tic:0.2f}s")

    if timing:
        tic = time.perf_counter()
    dominant = centroid[np.argmax(counts)]
    if timing:
        toc = time.perf_counter()
        print(f"Dominant selection in {toc - tic:0.2f}s")

    if timing:
        end = time.perf_counter()
        total_time = end - start
        print(f"cv2_dominant_colour execution in {total_time:0.2f}s")

    return dominant, labels, centroid, total_time


def sklearn_dominant_colour(img_url,  colours=10, timing=False):
    '''
    Dominant Colour method using sklearn, based on:
    https://medium.com/analytics-vidhya/colour-separation-in-an-image-using-kmeans-clustering-using-python-f994fa398454
    '''
    if timing:
        start = time.perf_counter()
        tic = time.perf_counter()
    img = io.imread(img_url)
    img = img.reshape((-1, 3))
    if timing:
        toc = time.perf_counter()
        print(f"Loaded the image in {toc - tic:0.2f}s")

    if timing:
        tic = time.perf_counter()
    cluster = KMeans(n_clusters=colours)
    cluster.fit(img)
    if timing:
        toc = time.perf_counter()
        print(f"KMeans calculation in {toc - tic:0.2f}s")

    labels = cluster.labels_
    labels = list(labels)
    centroid = cluster.cluster_centers_

    if timing:
        tic = time.perf_counter()
    percent = []

    for i in range(len(centroid)):
        j = labels.count(i)
        j = j/(len(labels))
        percent.append(j)
    if timing:
        toc = time.perf_counter()
        print(f"Percentage calculation in {toc - tic:0.2f}s")

    indices = np.argsort(percent)[::-1]
    dominant = centroid[indices[0]]
    if timing:
        end = time.perf_counter()
        total_time = end - start
        print(
            f"sklearn_dominant_colour execution in {total_time:0.2f}s")
    return dominant, labels, centroid, total_time


def fast_dominant_colour(img_url, colours=10, timing=False, scale=1.0):
    '''
    Faster method for web use that speeds up the sklearn variant.
    Also can use a scaling factor to improve the speed at cost of
    accuracy
    '''
    if timing:
        start = time.perf_counter()
        tic = time.perf_counter()
    img = io.imread(img_url)
    if scale != 1.0:
        img = rescale(img, scale, multichannel=True)
        img = img * 255
    img = img.reshape((-1, 3))
    if timing:
        toc = time.perf_counter()
        print(f"Loaded the image in {toc - tic:0.2f}s")

    if timing:
        tic = time.perf_counter()
    cluster = KMeans(n_clusters=colours, n_init=3, max_iter=10, tol=0.001)
    cluster.fit(img)
    if timing:
        toc = time.perf_counter()
        print(f"KMeans calculation in {toc - tic:0.2f}s")

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
        print(f"Percentage calculation in {toc - tic:0.2f}s")

    indices = np.argsort(percent)[::-1]
    dominant = centroid[indices[0]]
    if timing:
        end = time.perf_counter()
        total_time = end - start
        print(f"fast_dominant_colour execution in {total_time:0.2f}s")

    return dominant, labels, centroid, total_time


def visualise_colours(labels, centroids):
    '''
    Generate a visualisation of the colours in an image
    '''
    # Get the number of different clusters, create histogram, and normalise
    sorted_labels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins=sorted_labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    # Create frequency rect and iterate through each cluster's colour
    # and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colours = sorted(zip(hist, centroids))
    start = 0
    for (percent, colour) in colours:
        print(f"[{clamp(colour[0])}, {clamp(colour[0])}, {clamp(colour[0])}] ",
              "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50),
                      colour.astype("uint8").tolist(), -1)
        start = end
    return rect


def clamp(x):
    '''
    Utility function to return ints from 0-255
    '''
    return int(max(0, min(x, 255)))


def get_rgb_colour(img_url, debug=False):
    '''
    Method to print hex sting and return an rgb tuple of the
    dominant colour in an image
    '''
    dominant_colour = fast_dominant_colour(img_url, scale=0.1)
    r = dominant_colour[0]
    g = dominant_colour[1]
    b = dominant_colour[2]

    if debug:
        hex_str = "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))
        print(f'{hex_str}')

    rgb_colour = (clamp(r), clamp(g), clamp(b))

    return rgb_colour
