import cv2
import numpy as np

image_path = 'YASS.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from path '{image_path}'")
    exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

height, width = cleaned.shape
mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(mask, (1, 1), (width-2, height-2), 255, -1)

masked_image = cv2.bitwise_and(cleaned, mask)

contours, _ = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"Contours found: {len(contours)}")

debug_image = image.copy()
cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contours Debug", debug_image)
cv2.imshow("Masked Image", masked_image)
cv2.imshow("Threshold Image", binary_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]

if len(filtered_contours) == 0:
    print("No contours after filtering based on area.")
else:
    print(f"Found {len(filtered_contours)} contours after filtering.")

output_file = "track_coordinates.txt"
with open(output_file, "w") as file:
    for contour in filtered_contours:
        for point in contour:
            x, y = point[0]
            file.write(f"{x}, {y}\n")

    print(f"Saved track coordinates to '{output_file}'")

final_image = image.copy()
cv2.drawContours(final_image, filtered_contours, -1, (0, 255, 0), 2)
cv2.imshow("Final Contours", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

input_file = "track_coordinates.txt"
output_file = "resized_track_coordinates.txt"

original_width = 1200
original_height = 800
target_width = 1600
target_height = 800

y_scale = target_height / original_height

with open(input_file, "r") as infile:
    lines = infile.readlines()

with open(output_file, "w") as outfile:
    for line in lines:
        x, y = map(int, line.strip().split(", "))

        resized_x = x
        resized_y = int(y * y_scale)

        outfile.write(f"{resized_x}, {resized_y}\n")

print(f"Resized coordinates saved to '{output_file}'")