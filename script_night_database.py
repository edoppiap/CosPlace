import cv2
import os
import random

def convert_to_night(image):
    # Converte l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Riduci la luminosità dell'immagine per farla sembrare notturna
    darker = cv2.convertScaleAbs(gray, alpha=0.3, beta=0)

    return darker

input_dirs = ['/content/data/tokyo_xs/test/database', '/content/data/tokyo_xs/test/queries']
output_dirs = [
    '/content/data/tokyo_xs/day_database/database',
    '/content/data/tokyo_xs/day_database/queries',
    '/content/data/tokyo_xs/night_database/database',
    '/content/data/tokyo_xs/night_database/queries'
]

# Percentuale di immagini da convertire in notturne e da mantenere diurne
percentage = 0.3

for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for i in range(len(input_dirs)):
    input_dir = input_dirs[i]
    output_dir_day = output_dirs[i * 2]
    output_dir_night = output_dirs[i * 2 + 1]

    all_filenames = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Seleziona un sottoinsieme casuale di immagini da convertire in notturne
    night_filenames = random.sample(all_filenames, int(percentage * len(all_filenames)))

    # Seleziona un sottoinsieme casuale di immagini da mantenere diurne
    day_filenames = random.sample(all_filenames, int(percentage * len(all_filenames)))

    for filename in all_filenames:
        img = cv2.imread(os.path.join(input_dir, filename))

        if filename in night_filenames:
            # Converte l'immagine in notturna
            img_night = convert_to_night(img)
            cv2.imwrite(os.path.join(output_dir_night, filename), img_night)
        elif filename in day_filenames:
            # Copia l'immagine nel database diurno
            cv2.imwrite(os.path.join(output_dir_day, filename), img)
