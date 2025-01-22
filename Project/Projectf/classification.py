import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Label, Button
import webbrowser

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

# Load shirt dataset from file
def load_shirt_dataset(file_path):
    return pd.read_csv(file_path)

# Function to normalize and compare color strings
def normalize_color(color):
    return str(color).strip().lower()

# Function to open a link in the browser
def open_link(url):
    webbrowser.open(url)

# Function to display shirt images with URLs in a Tkinter window
def display_shirt_images_with_url(recommended_colors, shirt_data):
    normalized_colors = [normalize_color(color) for color in recommended_colors]
    shirt_data['Normalized_Color'] = shirt_data.iloc[:, 11].apply(normalize_color)

    # Filter shirts by recommended colors
    recommended_shirts = shirt_data[shirt_data['Normalized_Color'].isin(normalized_colors)]

    if recommended_shirts.empty:
        print("No shirts found matching the recommended colors.")
        return

    # Tkinter setup
    root = tk.Tk()
    root.title("Shirt Recommendations")

    for _, row in recommended_shirts.iterrows():
        img_path = row.iloc[5]
        product_url = row.iloc[10]

        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()

            # Add button to open product URL
            btn = Button(root, text="View Product", command=lambda url=product_url: open_link(url))
            btn.pack()
        except Exception as e:
            print(f"Error displaying image: {e}")

    root.mainloop()

# Example usage
skin_tone = "Medium Brown"  # Replace with detected skin tone
dataset_path = r"D:\KLH\SEM3\AIML\Project\Projectf\men-formal-shirts.csv\men_formal_shirts.csv"
shirt_data = load_shirt_dataset(dataset_path)
recommended_colors = recommend_clothing(skin_tone)
display_shirt_images_with_url(recommended_colors, shirt_data)
