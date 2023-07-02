import cv2
import os
import numpy as np

input_dirs = ['/content/data/tokyo_xs/test/database', '/content/data/tokyo_xs/test/queries']
output_dirs = ['/content/data/tokyo_xs/night_database/database', '/content/data/tokyo_xs/night_database/queries']

# Create the output directories if they don't exist
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Brightness threshold to determine if an image is nighttime
# This value may need to be adjusted based on your images
brightness_threshold = 80

for i in range(len(input_dirs)):
    input_dir = input_dirs[i]
    output_dir = output_dirs[i]

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read the image in grayscale
            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

            # Calculate the average brightness of the image
            brightness = np.mean(img)

            # If the image is below the brightness threshold, consider it nighttime
            if brightness < brightness_threshold:
                # Copy the image to the output directory
                cv2.imwrite(os.path.join(output_dir, filename), img)

print("Correctly generated datasets:")
for dir in output_dirs:
    print(f"Output directory: {dir}")
    print(f"Files in directory: {os.listdir(dir)}")
