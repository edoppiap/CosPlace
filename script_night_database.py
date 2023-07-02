import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def calculate_brightness(image):
    # Calcola la luminosità dell'immagine utilizzando KMeans per trovare i colori più comuni
    reshaped_image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(reshaped_image)
    most_common_colors = kmeans.cluster_centers_

    # Calcola la luminosità media dei colori più comuni
    brightness = np.mean(most_common_colors)

    return brightness

input_dirs = ['/content/data/tokyo_xs/test/database', '/content/data/tokyo_xs/test/queries']
output_dirs = [
    '/content/data/tokyo_xs/night_database/database',
    '/content/data/tokyo_xs/night_database/queries',
    '/content/data/tokyo_xs/day_database/database',
    '/content/data/tokyo_xs/day_database/queries'
]

for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

brightness_threshold_night = 60
brightness_threshold_day = 200

for i in range(len(input_dirs)):
    input_dir = input_dirs[i]
    output_dir_night = output_dirs[i * 2]
    output_dir_day = output_dirs[i * 2 + 1]

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(input_dir, filename))
            brightness = calculate_brightness(img)

            if brightness < brightness_threshold_night:
                cv2.imwrite(os.path.join(output_dir_night, filename), img)
            elif brightness > brightness_threshold_day:
                cv2.imwrite(os.path.join(output_dir_day, filename), img)
