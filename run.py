import os
import matplotlib.pyplot as plt
from dominant_colour import cv2_dominant_colour, fast_dominant_colour, skimage_dominant_colour, visualise_colours
from skimage import io

colour_list = [5, 10, 15]
# Image 1: 6 Nimmt, mostly yellow; Image 2: Altiplano, a wide gamut of colours
img_urls = {'6 Nimmt': 'https://images-na.ssl-images-amazon.com/images/I/81HmoJNlUnL._AC_SL1500_.jpg',
            'Altiplano': 'https://cf.geekdo-images.com/hgUDu_oG0uhnOWX4WM2vXA__imagepage/img/dDNwKBxSKDViEa1EcHh1QiM8cM4=/fit-in/900x600/filters:no_upscale():strip_icc()/pic4070329.jpg'}

if not os.path.exists('figures'):
    print('Creating figure dir')
    os.makedirs('figures')

for name, img_url in img_urls.items():
    for colours in colour_list:
        # Call each method to determine dominant colour
        dominant_cv2, labels_cv2, centroids_cv2, time_cv2 = cv2_dominant_colour(
            img_url, colours=colours, timing=True)
        dominant_skimage, labels_skimage, centroids_skimage, time_skimage = skimage_dominant_colour(
            img_url, colours=colours, timing=True)
        dominant_fast, labels_fast, centroids_fast, time_fast = fast_dominant_colour(
            img_url, colours=colours, timing=True)
        dominant_faster, labels_faster, centroids_faster, time_faster = fast_dominant_colour(
            img_url, colours=colours, timing=True, scale=0.1)

        # Return image of most dominant colours in histogram
        visualise_cv2 = visualise_colours(labels_cv2, centroids_cv2)
        visualise_skimage = visualise_colours(
            labels_skimage, centroids_skimage)
        visualise_fast = visualise_colours(labels_fast, centroids_fast)
        visualise_faster = visualise_colours(labels_faster, centroids_faster)

        # Plot images next to colour bars
        fig = plt.figure()
        ax = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
        img = io.imread(img_url)
        ax.imshow(img)
        ax.axis('off')
        ax0 = plt.subplot2grid((4, 2), (0, 1))
        ax0.imshow(visualise_cv2)
        ax0.set_title(f'CV2 Method in {time_cv2:0.2f}s')
        ax0.axis('off')
        ax1 = plt.subplot2grid((4, 2), (1, 1))
        ax1.imshow(visualise_skimage)
        ax1.set_title(f'Skimage Method in {time_skimage:0.2f}s')
        ax1.axis('off')
        ax2 = plt.subplot2grid((4, 2), (2, 1))
        ax2.imshow(visualise_fast)
        ax2.set_title(f'Fast Method in {time_fast:0.2f}s')
        ax2.axis('off')
        ax3 = plt.subplot2grid((4, 2), (3, 1))
        ax3.imshow(visualise_faster)
        ax3.set_title(f'Faster Method in {time_faster:0.2f}s')
        ax3.axis('off')
        fig.suptitle(f'{name}: {colours} Colour Clusters', fontsize=16)
        fig.canvas.draw()
        plt.savefig(f'figures/{name}_{str(colours)}.png')
        plt.pause(0.001)
input('Press Enter to continue...')
