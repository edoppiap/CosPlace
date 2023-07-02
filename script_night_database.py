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

# Create the output directories if they don't exist
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Brightness thresholds to determine if an image is nighttime or daytime
night_brightness_threshold = 80
day_brightness_threshold = 120

for i in range(len(input_dirs)):
    input_dir = input_dirs[i]
    output_dir = output_dirs[i * 2]  # Output directory for nighttime images
    output_dir_day = output_dirs[i * 2 + 1]  # Output directory for daytime images

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read the image in grayscale
            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

            # Calculate the average brightness of the image
            brightness = np.mean(img)

            # If the image is below the brightness threshold, consider it nighttime
            if brightness < night_brightness_threshold:
                # Copy the image to the nighttime output directory
                cv2.imwrite(os.path.join(output_dir, filename), img)
            # If the image is above the day brightness threshold, consider it daytime
            elif brightness > day_brightness_threshold:
                # Copy the image to the daytime output directory
                cv2.imwrite(os.path.join(output_dir_day, filename), img)

print("Correctly generated datasets:")
for i in range(len(output_dirs)):
    dir = output_dirs[i]
    print(f"Output directory: {dir}")
    print(f"Files in directory: {os.listdir(dir)}")
