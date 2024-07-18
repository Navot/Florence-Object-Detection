from flask import Flask, request, jsonify, render_template_string, send_from_directory
from unittest.mock import patch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports
import os
from typing import List, Union
from io import BytesIO
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

app = Flask(__name__)

# Create a directory for temporary images if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create a directory for temporary images if it doesn't exist
UPLOAD_FOLDER = os.path.join(script_dir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fix for importing modules in Florence-2 model
def fixed_get_imports(filename: Union[str, os.PathLike]) -> List[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

# Apply the patch to fix imports
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

# Function to run the model and process the image and prompt
def run_example(prompt, image):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    return parsed_answer

@app.route('/', methods=['GET', 'POST'])
def index():
    template_path = os.path.join(script_dir, 'index.html')
    with open(template_path, 'r') as file:
        html_template = file.read()
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        prompt = request.form['prompt']
        get_box_objects = 'get_box_objects' in request.form
        image = None
        local_image_path = None
        if 'image_file' in request.files and request.files['image_file'].filename != '':
            logging.info("Image File Route")
            file = request.files['image_file']
            logging.info(file)
            local_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            logging.info("local_image_path: " + local_image_path)
            file.save(local_image_path)
            image = Image.open(local_image_path)
        elif image_url:
            image = Image.open(requests.get(image_url, stream=True).raw)
        
        if image:
            parsed_answer = run_example(prompt, image)
            coordinates = {}
            if get_box_objects:
                coordinates = run_example("<OD>", image)  # Object detection call
                logging.info(f"Coordinates: {coordinates}")
                new_image = plot_bbox(image, coordinates.get("<OD>", {}))  # Safely access <OD>
                new_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'new_' + (os.path.basename(local_image_path)))
                logging.info(f"Saving new image with bounding boxes to: {new_image_path}")
                new_image.save(new_image_path)
                local_image_path = new_image_path  # Update to the new image path
            return render_template_string(html_template, parsed_answer=parsed_answer, coordinates=coordinates, image_url=image_url, local_image_path=local_image_path, prompt=prompt)
    return render_template_string(html_template, parsed_answer=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/generate-caption-url', methods=['POST'])
def generate_caption_url():
    data = request.get_json()
    image_url = data.get('image_url')
    prompt = data.get('prompt')
    
    if not image_url or not prompt:
        return jsonify({'error': 'Image URL and prompt are required.'}), 400
    
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
        parsed_answer = run_example(prompt, image)
        return jsonify({'parsed_answer': parsed_answer})
    except Exception as e:
        logging.error(f"Error processing image from URL: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-caption-file', methods=['POST'])
def generate_caption_file():
    if 'image_file' not in request.files or not request.form.get('prompt'):
        return jsonify({'error': 'Image file and prompt are required.'}), 400
    
    file = request.files['image_file']
    prompt = request.form['prompt']
    
    try:
        image = Image.open(file.stream)
        parsed_answer = run_example(prompt, image)
        return jsonify({'parsed_answer': parsed_answer})
    except Exception as e:
        logging.error(f"Error processing uploaded image: {e}")
        return jsonify({'error': str(e)}), 500

def plot_bbox(image, data):
    # Debugging: print the structure of data
    print("Data received by plot_bbox:", data)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Ensure data contains expected keys
    if 'bboxes' not in data or 'labels' not in data:
        raise KeyError("Expected keys 'bboxes' and 'labels' not found in data")

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Create an Image object from the buffer
    result_image = Image.open(buf)

    # Convert image to RGB if it's in RGBA mode
    if result_image.mode == 'RGBA':
        result_image = result_image.convert('RGB')

    return result_image

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
