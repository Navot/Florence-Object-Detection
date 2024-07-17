# Florence-Object-Detection
Image Captioning Flask App - This Flask application provides a web interface and API endpoints for generating image captions and detecting objects in images using a pre-trained model - Florence-2-large from HaggingFaces.

## Features:
Upload images or provide image URLs to generate captions.
Detect objects in images and draw bounding boxes around them.
API endpoints for generating captions from image URLs or uploaded image files.

## Setup and Installation
Prerequisites:
Python 3.7+
Pip (Python package installer)

## Install Dependencies
pip install -r requirements.txt

## Run the Application
python Florence-2-large-Object-Detection.py
The application will be available at http://127.0.0.1:5000

## API Endpoints
### Generate Caption from URL
Endpoint: /api/generate-caption-url
Method: POST
Payload:
{
  "image_url": "URL of the image",
  "prompt": "Text prompt for caption generation"
}
Response:
{
  "parsed_answer": "Generated caption"
}

### Generate Caption from Uploaded File
Endpoint: /api/generate-caption-file
Method: POST
Payload:
{
  "image_path": "Path to the image file",
  "prompt": "Text prompt for caption generation"
}
Response:
{
  "parsed_answer": "Generated caption"
}

## Web Interface
Access the web interface at http://127.0.0.1:5000/
Upload an image or provide an image URL.
Enter a text prompt and click "Generate Caption" to get the generated caption and see the image with bounding boxes for detected objects.

## Logging
Logging is configured to capture detailed information about the application's operations. Check the console output for logs.

## Troubleshooting
Ensure all dependencies are installed correctly.
Verify that the image URLs provided are accessible and correct.
Check the application logs for detailed error messages.
## License
This project is licensed under the MIT License. See the LICENSE file for details.

