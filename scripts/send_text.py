import requests
import os
from dotenv import load_dotenv
load_dotenv()

URL = os.getenv("N8N_TEXT_URL")
KEY = os.getenv("N8N_TEXT_KEY")

def send_text(message: str):    
    if not message:
        raise ValueError("message is required")
    if not URL or not KEY:
        raise ValueError("url and key are required")
    
    url = URL
    key = KEY
    
    payload = message
    headers = {
    'Authorization': f"Bearer {key}",
    'Content-Type': 'text/plain'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
    
    print(f"Sent message: {message}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Send a text message via n8n.")
    parser.add_argument("message", type=str, help="The message to send")
    args = parser.parse_args()
    
    send_text(args.message)