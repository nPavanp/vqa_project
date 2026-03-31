import requests
import base64
import os

# ✅ Use new Hugging Face router endpoint (IMPORTANT)
API_URL = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-vqa-base"

# ✅ Secure token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def predict(image_path, question):
    try:
        # Read image
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # Encode image to base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "inputs": {
                "image": img_base64,
                "question": question
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        # Debug print (helps on Render logs)
        print("HF Response:", response.text)

        if response.status_code != 200:
            return f"Error: {response.text}"

        result = response.json()

        # Handle different response formats
        if isinstance(result, list):
            return result[0].get("generated_text", "No answer found")

        return str(result)

    except Exception as e:
        print("ERROR:", str(e))
        return f"Error: {str(e)}"