import keras_ocr
import dateparser
import re
import matplotlib.pyplot as plt

# Function to detect and recognize text in an image
def detect_text(image_path):
    # Load the image
    image = keras_ocr.tools.read(image_path)
    
    # Initialize Keras-OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()
    
    # Perform text detection and recognition
    prediction_groups = pipeline.recognize([image])
    
    return prediction_groups[0]  # Return text from first (and only) image

# Function to filter out non-date text
def filter_date_texts(texts):
    date_texts = []
    date_pattern = r'\b\d{4}\d{2}\d{2}\b'  # yyyymmdd format
    for text, box in texts:
        if re.search(date_pattern, text):
            date_texts.append((text, box))
    return date_texts

# Function to extract dates from recognized text
def extract_dates(texts):
    dates = []
    for text, box in texts:
        # Ensure the date is in the yyyymmdd format
        if len(text) == 8 and text.isdigit():
            try:
                parsed_date = dateparser.parse(text)
                if parsed_date:
                    dates.append((text, parsed_date))
            except ValueError:
                continue
    return dates

# Function to plot the image with recognized text
def plot_predictions(image_path, predictions):
    # Load the image
    image = keras_ocr.tools.read(image_path)
    
    # Plot the image
    plt.imshow(image)
    
    # Plot the recognized text boxes
    for text, box in predictions:
        box = box.astype(int)
        plt.plot([box[0][0], box[1][0], box[2][0], box[3][0], box[0][0]],
                 [box[0][1], box[1][1], box[2][1], box[3][1], box[0][1]], 
                 linewidth=2)
        plt.text(box[0][0], box[0][1] - 10, text, color='red', fontsize=12, 
                 backgroundcolor='white')
    
    plt.axis('off')
    plt.show()

# Path to your image
image_path = 'cubs.PNG'

# Detect and recognize text in the image
recognized_texts = detect_text(image_path)

# Filter and extract date-like texts
date_texts = filter_date_texts(recognized_texts)

# Extract dates from recognized text
dates = extract_dates(date_texts)

# Print the extracted dates
for original_text, parsed_date in dates:
    print(f"Original Text: {original_text} -> Parsed Date: {parsed_date}")

# Display the image with recognized text
plot_predictions(image_path, date_texts)
