import matplotlib.pyplot as plt
import keras_ocr

# keras-ocr will automatically download pretrained weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Define the path to your local image
image_paths = [
    'BTC.png'  # Replace with your image path
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
        # Filter predictions to keep only "TD", "BUY", and "SELL" labels
        filtered_predictions = [prediction for prediction in predictions if prediction[0].upper() in ["TD", "BUY", "SELL"]]

        # Sort the filtered predictions based on their x-coordinates (right to left)
        filtered_predictions.sort(key=lambda x: x[1][0][0], reverse=True)

        # Keep the rightmost "TD" prediction and the rightmost "BUY" or "SELL" prediction
        rightmost_td_prediction = None
        rightmost_buy_sell_prediction = None

        for prediction in filtered_predictions:
            label = prediction[0].upper()
            if label == "TD" and rightmost_td_prediction is None:
                rightmost_td_prediction = prediction
            elif label in ["BUY", "SELL"] and rightmost_buy_sell_prediction is None:
                rightmost_buy_sell_prediction = prediction

        # Check if the rightmost "TD" prediction is vertically aligned with the rightmost "BUY" or "SELL" prediction
        if rightmost_td_prediction and rightmost_buy_sell_prediction:
            td_y = rightmost_td_prediction[1][0][1]
            buy_sell_y = rightmost_buy_sell_prediction[1][0][1]
            if abs(td_y - buy_sell_y) < 10:  # Adjust the threshold as needed
                print(f"TD and {rightmost_buy_sell_prediction[0]}")
            else:
                print("None")
        else:
            print("None")

        # Draw annotations for the rightmost predictions
        predictions_to_draw = []
        if rightmost_td_prediction:
            predictions_to_draw.append(rightmost_td_prediction)
        if rightmost_buy_sell_prediction:
            predictions_to_draw.append(rightmost_buy_sell_prediction)

        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions_to_draw, ax=ax)

    plt.show()
else:
    print("No images to process.")