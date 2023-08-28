import csv
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
from collections import defaultdict
import pickle
import torch
import torch.nn as nn
import pandas as pd
import json
# from io import BytesIO

app = Flask(__name__)

model_path = "/home/python/Aachal/Image_colour_name_detection/color_classifier.pth"
label_encoder_path = "/home/python/Aachal/Image_colour_name_detection/label_encoder.pkl"
csv_file = "/home/python/Aachal/Image_colour_name_detection/colors (2).csv"


def get_image_dimensions(image):
    width, height = image.size
    ppi = image.info.get('dpi', (72.0, 72.0))
    return width, height, ppi


def pixels_to_inches(pixels, ppi):
    inches = pixels / ppi
    return inches

# Define the ColorClassifier model
class ColorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ColorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to get color information from an image


def get_image_colors(image, model, label_encoder):
    image = Image.open(image)
    image = image.convert('RGB')
    pixels = np.array(image.getdata()) / 255.0
    pixels = torch.tensor(pixels, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        outputs = model(pixels)
        _, predicted_indices = torch.max(outputs, 1)
        color_names = label_encoder.inverse_transform(
            predicted_indices.numpy())
    color_counter = defaultdict(int)
    for color_name in color_names:
        color_counter[color_name] += 1
    return color_counter


def get_rgb_from_csv(colors_list, csv_file_path):
    colors_dict = {}

    try:
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Check if "Name" exists in fieldnames
            if "Name" not in reader.fieldnames:
                raise ValueError("CSV file does not have a 'Name' column.")

            for row in reader:
                # Check if "Name" exists in the row
                if "Name" in row:
                    color_name = row["Name"].lower()
                    rgb = (
                        int(row.get("Red (8 bit)", 0)),
                        int(row.get("Green (8 bit)", 0)),
                        int(row.get("Blue (8 bit)", 0))
                    )
                    colors_dict[color_name] = rgb
                else:
                    # Handle the case where "Name" is missing in the row
                    # You can log this or take appropriate action
                    pass

    except FileNotFoundError:
        # Handle the case where the CSV file is not found
        raise FileNotFoundError("CSV file not found.")
    except Exception as e:
        # Handle other exceptions that may occur during file processing
        raise e

    result = []
    for color_name in colors_list:
        if color_name.lower() in colors_dict:
            rgb = colors_dict[color_name.lower()]
            result.append((color_name, rgb))

    return result


# Function to convert RGB to hex
def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

# Endpoint to handle image uploads and return results in JSON format


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})

        image = request.files['image']
        # image = Image.open(BytesIO(image_file))

        num_classes = 0
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
            num_classes = len(label_encoder.classes_)

        model = ColorClassifier(3, num_classes)
        model.load_state_dict(torch.load(model_path))

        image_colors = get_image_colors(image, model, label_encoder)
        color_pixels = sorted(image_colors.items(),
                              key=lambda x: x[1], reverse=True)[:10]
        color_codes = get_rgb_from_csv([color[0]
                                        for color in color_pixels], csv_file)

        result_data = []
        for (color_name, rgb_val), (_, pixels) in zip(color_codes, color_pixels):
            hex_value = rgb_to_hex(rgb_val)      
            result_data.append({
                'color_name': color_name,
                # 'rgb_value': rgb_val,
                'hex_value': hex_value,
                'pixels': pixels
            })

        width, height, ppi = get_image_dimensions(Image.open(image))
        in_inch = f"{round(pixels_to_inches(width, ppi[0]), 2)} inch x {round(pixels_to_inches(height, ppi[1]), 2)} inch"
        # print(f"DPI: {dpi_x} x {dpi_y}")
        # print(f"Print size: {print_width:.2f} inches x {print_height:.2f} inches")

        if width is not None and height is not None and ppi is not None:
            in_pixel = f"{width} pixel x {height} pixel"
            result = {
                "width": width,
                "height": height,
                "pixel": in_pixel,
                "inches": in_inch,
                # "width_inches": round(pixels_to_inches(width, ppi[0]), 2),
                # "height_inches": round(pixels_to_inches(height, ppi[1]), 2),
                "total_resolution": width * height
            }

        # return jsonify({'results': result_data, 'image_dimensions': result})
        return json.dumps({'Image_Colour_Properties': result_data, 'image_dimensions': result}, indent=10)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
                                        

if __name__ == '__main__':
    app.run(debug=True)
