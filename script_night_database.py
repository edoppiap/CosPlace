import cv2
import os
import numpy as np

input_dir = '/kaggle/working/data/tokyo_xs/database'
output_dir = '/kaggle/working/data/tokyo_xs/night_database'

# Creare la directory di output se non esiste
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Soglia di luminosità per determinare se un'immagine è notturna
brightness_threshold = 80

for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Leggere l'immagine in scala di grigi
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Calcolare la luminosità media dell'immagine
        brightness = np.mean(img)

        # Se l'immagine è sotto la soglia di luminosità, la consideriamo notturna
        if brightness < brightness_threshold:
            # Copiare l'immagine nella directory di output
            cv2.imwrite(os.path.join(output_dir, filename), img)
