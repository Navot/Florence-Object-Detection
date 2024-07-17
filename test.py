import requests
import json

def send_post_request(api_url, json_payload):
    headers = {
        'Content-Type': 'application/json; utf-8',
        'Accept': 'application/json'
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(json_payload))
    
    if response.status_code == 200:
        json_response = response.json()
        parsed_answer = json_response.get("parsed_answer", {})
        return json.dumps(parsed_answer, indent=4)
    else:
        return f"Request failed with response code: {response.status_code}"

def generate_caption_from_url(api_url, image_url, prompt):
    try:
        json_payload = {
            "image_url": image_url,
            "prompt": prompt
        }
        return send_post_request(api_url, json_payload)
    except Exception as e:
        return f"Exception: {str(e)}"

def generate_caption_from_file(api_url, image_path, prompt):
    try:
        json_payload = {
            "image_path": image_path,
            "prompt": prompt
        }
        return send_post_request(api_url, json_payload)
    except Exception as e:
        return f"Exception: {str(e)}"

if __name__ == "__main__":
    print("Starting")
    api_url_for_url = "http://127.0.0.1:5000/api/generate-caption-url"
    api_url_for_file = "http://127.0.0.1:5000/api/generate-caption-file"
    image_url = "https://media.licdn.com/dms/image/C4D16AQFP9-46LMUGfg/profile-displaybackgroundimage-shrink_350_1400/0/1642552059040?e=1726704000&v=beta&t=U4VKOyWOXVu0j1eQorzt49KiRiLpJya4mMUkU2eNICE"
    local_image_path = "/Users/navotdako/Dev/tpg/data/gratisography-holographic-suit-1170x780.jpg"
    prompt = "<MORE_DETAILED_CAPTION>"
    
    caption_from_url = generate_caption_from_url(api_url_for_url, image_url, prompt)
    print("Generated Caption from URL: " + caption_from_url)
