import cv2

# Read image
image = cv2.imread("image.jpg")

# Check if image loaded
if image is None:
    print("Image not found!")
    exit()

# Display original image
cv2.imshow("Original Image", image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display grayscale image
cv2.imshow("Grayscale Image", gray)

# Save grayscale image
cv2.imwrite("gray_image.jpg", gray)

print("Press any key to close the windows...")

# Wait for key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
