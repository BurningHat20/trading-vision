import os
import cv2
import datetime
import keras_ocr

# keras-ocr will automatically download pretrained weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Define paths
image_path = 'india.png'
base_folder = 'D:/today'  # Update to your desired base folder path

# Load and preprocess the image
def resize_image(image, max_size=1024):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scaling_factor = max_size / float(max(height, width))
        return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

# Load the image
image_data = cv2.imread(image_path)
if image_data is None:
    print(f"Error: Could not load image at path: {image_path}")
else:
    image_data = resize_image(image_data)

    # Recognize text in the image
    predictions = pipeline.recognize([image_data])

    # Filter predictions
    buy_predictions = [p for p in predictions[0] if p[0].upper() == 'BUY']
    sell_predictions = [p for p in predictions[0] if p[0].upper() == 'SELL']

    # Determine the folder path based on the current date
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    folder_path = os.path.join(base_folder, current_date)

    # Create folders if they don't exist
    buy_folder = os.path.join(folder_path, 'Buy')
    sell_folder = os.path.join(folder_path, 'Sell')
    os.makedirs(buy_folder, exist_ok=True)
    os.makedirs(sell_folder, exist_ok=True)

    # Save the image in the appropriate folder
    if buy_predictions:
        cv2.imwrite(os.path.join(buy_folder, 'india.png'), image_data)
        print("Image saved in Buy folder.")
    elif sell_predictions:
        cv2.imwrite(os.path.join(sell_folder, 'india.png'), image_data)
        print("Image saved in Sell folder.")
    else:
        print("No BUY or SELL signal detected.")
