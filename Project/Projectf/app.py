from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import cv2
import numpy as np
import imutils
from sklearn.cluster import KMeans
import os
from werkzeug.utils import secure_filename #Secure File Upload

app = Flask(__name__)

# Example CSV path (change to your own file path)
dataset_file = r'D:\KLH\SEM3\AIML\Project\Projectf\men-formal-shirts.csv\men_formal_shirts.csv'  # Update this path

# Function to extract skin area (using simple HSV thresholding)
def extractSkin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin

# Function to extract dominant colors (using KMeans clustering)
def extractDominantColor(image, k=2, hasThresholding=False):
    if hasThresholding:
        image = extractSkin(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    return dominant_colors

# Function to determine the skin tone based on dominant color values
def determine_skin_tone(dominant_colors):
    color_value = dominant_colors[0]
    r, g, b = color_value[0], color_value[1], color_value[2]
    
    # Adjusted skin tone detection based on refined RGB values
    if r > 200 and g > 170 and b > 150:
        return "Fair"
    elif r > 170 and g > 140 and b > 130:
        return "Wheatish"
    elif r > 150 and g > 120 and b > 90:
        return "Medium Brown"
    elif r > 130 and g > 110 and b > 80:
        return "Brown"
    elif r > 110 and g > 90 and b > 60:
        return "Dark Brown"
    #else:
        return "Very Dark Brown"


# Function to detect skin tone from an uploaded image
def detect_skin_tone(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image {image_path} could not be loaded.")
            return None

        print(f"Processing image: {image_path}")
        print(f"Original image shape: {image.shape}")

        image_resized = imutils.resize(image, width=250)
        print(f"Resized image shape: {image_resized.shape}")

        skin = extractSkin(image_resized)
        if skin is None or np.sum(skin) == 0:
            print(f"Failed to detect skin in {image_path}")
            return None

        dominant_colors = extractDominantColor(image_resized, hasThresholding=True)
        if dominant_colors is None:
            print(f"No dominant colors detected in {image_path}")
            return None

        skin_tone = determine_skin_tone(dominant_colors)
        print(f"Detected skin tone: {skin_tone}")
        return skin_tone

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# Function to recommend clothing based on skin tone
def recommend_clothing(skin_tone):
    clothing_recommendations = {
        "Fair": ["green", "blue", "black", "red", "navy"],
        "Wheatish": ["green", "yellow", "white", "purple", "grey", "navy", "lime green"],
        "Medium Brown": ["light grey", "blue", "lavender", "green", "purple", "cream"],
        "Brown": ["burgundy", "dark grey", "green", "maroon", "navy blue", "white"],
        "Dark Brown": ["white", "green", "purple", "dark grey", "mustard", "coffee brown", "charcoal grey"]
    }
    return clothing_recommendations.get(skin_tone, [])

# Load shirt dataset
def load_shirt_dataset(file_path):
    return pd.read_csv(file_path)

# Normalize color function
def normalize_color(color):
    return str(color).strip().lower()

# Function to display shirt images based on recommended colors and show product URL
def display_shirt_images_with_url(recommended_colors, shirt_data, start_index=0, show_more=10):
    normalized_recommended_colors = [normalize_color(color) for color in recommended_colors]
    shirt_data['Normalized_Color'] = shirt_data.iloc[:, 11].apply(normalize_color)  # 12th column for colors
    recommended_shirts = shirt_data[shirt_data['Normalized_Color'].isin(normalized_recommended_colors)]

    if recommended_shirts.empty:
        print("No shirts found that match the recommended colors.")  # Debugging line
        return "No shirts found that match the recommended colors."

    subset_shirts = recommended_shirts.iloc[start_index:start_index + show_more]
    images = []

    for _, row in subset_shirts.iterrows():
        img_url = row.iloc[5]  # 6th column for image URLs
        product_url = row.iloc[10]  # 11th column for product URL

        # Check if the URL is valid and accessible
        try:
            response = requests.get(img_url)
            if response.status_code == 200:
                # Only append the img_url if it's valid
                images.append({"img_url": img_url, "product_url": product_url})
            else:
                print(f"Error loading image from URL: {img_url}, Status Code: {response.status_code}")
        except Exception as e:
            print(f"Unable to load image at URL: {img_url}. Error: {e}")

    print(f"Matching shirts: {images}")  # Debugging line to check if images are being added
    return images


# Home route
@app.route('/')
def home():
    return render_template('Home.html')

# Image upload and skin tone detection route
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_image():
    image = request.files['image']
    
    if image:
        # Save uploaded image to the static/uploads folder
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Detect skin tone
        detected_skin_tone = detect_skin_tone(image_path)

        if detected_skin_tone:
            # After detecting skin tone, redirect to the result page with the detected skin tone
            return redirect(url_for('result', skin_tone=detected_skin_tone, image_path=filename))
        else:
            return "Unable to detect skin tone. Please try again."

# Skin tone detection result page
@app.route('/result')
def result():
    skin_tone = request.args.get('skin_tone')
    image_path = request.args.get('image_path')  # Get the uploaded image filename

    # Get recommended colors based on skin tone
    recommended_colors = recommend_clothing(skin_tone)

    # Print recommended colors to the console for debugging
    print(f"Recommended colors for {skin_tone}: {recommended_colors}")

    return render_template('Detect_Skin_Tone.html', 
                           skin_tone=skin_tone, 
                           recommended_colors=recommended_colors, 
                           image_path=image_path)

# Recommendations route to display product images and URLs based on recommended colors
@app.route('/recommendations')
def recommendations():
    skin_tone = request.args.get('skin_tone')
    start_index = int(request.args.get('start_index', 0))  # Get start_index, default to 0
    recommended_colors = recommend_clothing(skin_tone)

    # Load dataset and filter shirts based on recommended colors
    shirt_data = load_shirt_dataset(dataset_file)
    matching_shirts = display_shirt_images_with_url(recommended_colors, shirt_data, start_index=start_index)

    # Return the recommendations as a rendered template fragment (not full HTML)
    return render_template('Recommendations.html', matching_shirts=matching_shirts)


if __name__ == '__main__':
    app.run(debug=True)
