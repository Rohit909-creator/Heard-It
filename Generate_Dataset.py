import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

def generate_wakewords(wakeword: str):
    # Cartesia API endpoint
    url = "https://api.cartesia.ai/tts/bytes"
    
    # Define voice IDs
    voice_ids = [
        "9b953e7b-86a8-42f0-b625-1434fb15392b",  # Original voice from your example
        "faf0731e-dfb9-4cfc-8119-259a79b27e12",  # Adding more example voice IDs
        "bec003e2-3cb3-429c-8468-206a393c67ad",
        "e61a659d-56d5-4023-a499-1e1bccbc40e9",
        "28ca2041-5dda-42df-8123-f58ea9c3da00",
        "d088cdf6-0ef0-4656-aea8-eb9b004e82eb",
        "fd2ada67-c2d9-4afe-b474-6386b87d8fc3"
    ]
    
    # Languages to try
    languages = ["hi"]
    
    # Headers
    headers = {
        "Cartesia-Version": "2024-06-10",
        "X-API-Key": os.getenv("API_KEY"),  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    
    for voice_id in voice_ids:
        for language in languages:
            # Prepare the request payload based on your example
            payload = {
                "model_id": "sonic",
                "transcript": wakeword,
                "voice": {
                    "mode": "id",
                    "id": voice_id,
                    "__experimental_controls": {
                        "speed": 0
                    }
                },
                "output_format": {
                    "container": "wav",
                    "encoding": "pcm_f32le",
                    "sample_rate": 16000
                },
                "language": language
            }
            
            # Make the API request
            try:
                response = requests.post(url, headers=headers, json=payload)
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Define output filename
                    output_filename = f"{wakeword}_{voice_id}_{language}.wav"
                    
                    # Save the audio bytes to a file
                    with open(output_filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Audio file created successfully: '{output_filename}'")
                else:
                    print(f"Failed to generate audio. Status code: {response.status_code}")
                    print(f"Response: {response.text}")
                
            except Exception as e:
                print(f"Error occurred: {str(e)}")
            
            # Add a small delay between requests to avoid rate limiting
            time.sleep(0.5)

# Example usage
# generate_wakewords("Hello World")

if __name__ == "__main__":
    
    with open("hindi_commands.txt", 'r') as f:
        s = f.read()
    words = s.split('\n')
    print(words)
    
    for word in words:
        generate_wakewords(word)
        time.sleep(2)
    
    for word in words:
        try:
            os.makedirs(f"./Audio_dataset/{word}", exist_ok=True)
        except Exception as e:
            print("Exception:", e)
            