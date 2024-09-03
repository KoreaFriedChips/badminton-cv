import cv2

# Load the image using OpenCV
image_path = 'input_videos/image.png'  # Replace with your actual image path
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Get the dimensions of the image
    original_h, original_w = image.shape[:2]

    # Print the dimensions
    print(f"Image width: {original_w}, Image height: {original_h}")
