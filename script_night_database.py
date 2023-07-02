import cv2
import os
import numpy as np

input_dirs = ['/content/data/tokyo_xs/test/database', '/content/data/tokyo_xs/test/queries']
output_dirs = [
    '/content/data/tokyo_xs/night_database/database',
    '/content/data/tokyo_xs/night_database/queries',
    '/content/data/tokyo_xs/day_database/database',
    '/content/data/tokyo_xs/day_database/queries'
]

# Creare le directory di output se non esistono
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Soglie di luminosità per determinare se un'immagine è notturna o diurna
night_brightness_threshold = 85
day_brightness_threshold = 170

for i in range(len(input_dirs)):
    input_dir = input_dirs[i]
    output_dir = output_dirs[i * 2]  # Directory di output per le immagini notturne
    output_dir_day = output_dirs[i * 2 + 1]  # Directory di output per le immagini diurne

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Leggere l'immagine a colori
            img = cv2.imread(os.path.join(input_dir, filename))

            # Calcolare la luminosità media dell'immagine
            brightness = np.mean(img)

            # Se l'immagine è al di sotto della soglia di luminosità, considerarla notturna
            if brightness < night_brightness_threshold:
                # Copiare l'immagine nella directory di output notturno
                cv2.imwrite(os.path.join(output_dir, filename), img)
            # Se l'immagine è sopra la soglia di luminosità diurna, considerarla diurna
            elif brightness > day_brightness_threshold:
                # Copiare l'immagine nella directory di output diurno
                cv2.imwrite(os.path.join(output_dir_day, filename), img)

print("Datasets generati correttamente:")
for i in range(len(output_dirs)):
    dir = output_dirs[i]
    print(f"Directory di output: {dir}")
    print(f"File nella directory: {os.listdir(dir)}")
