import matplotlib.pyplot as plt
import keras_ocr

# keras-ocr will automatically download pretrained weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Define the path to your local image
image_paths = [
    'cubs.png'  # Replace with your image path
]

# Load the images
images = [keras_ocr.tools.read(path) for path in image_paths]

# Perform text detection and recognition
if images:  # Proceed only if images were loaded
    prediction_groups = pipeline.recognize(images)

    # Create subplots
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))

    # Ensure axs is a list even if there is only one image
    if len(images) == 1:
        axs = [axs]

    # Plot the predictions
    for ax, image, predictions in zip(axs, images, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
    
    plt.show()
else:
    print("No images to process.")
