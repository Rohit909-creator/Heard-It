import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

def generate_wakewords(wakeword: str):
    # Cartesia API endpoint
    url = "https://api.cartesia.ai/tts/bytes"
    
    # Define voice IDs
    # voice_ids = [
    #     "9b953e7b-86a8-42f0-b625-1434fb15392b",  # Original voice from your example
    #     "faf0731e-dfb9-4cfc-8119-259a79b27e12",  # Adding more example voice IDs
    #     "bec003e2-3cb3-429c-8468-206a393c67ad",
    #     "e61a659d-56d5-4023-a499-1e1bccbc40e9",
    #     "28ca2041-5dda-42df-8123-f58ea9c3da00",
    #     "d088cdf6-0ef0-4656-aea8-eb9b004e82eb",
    #     "fd2ada67-c2d9-4afe-b474-6386b87d8fc3"
    # ]
    
    voice_ids = [
        "bf0a246a-8642-498a-9950-80c35e9276b5",
        "c99d36f3-5ffd-4253-803a-535c1bc9c306",
        "00967b2f-88a6-4a31-8153-110a92134b9f",
        "57c63422-d911-4666-815b-0c332e4d7d6a",
        "9fb269e7-70fe-4cbe-aa3f-28bdb67e3e84",
        
    ]
    # Languages to try
    languages = ["en"]
    
    # Headers
    headers = {
        "Cartesia-Version": "2024-06-10",
        "X-API-Key": os.getenv("API_KEY"),  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    count = 0
    for voice_id in voice_ids:
        count+=1
        print(count)
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
                    output_filename = f"./Audio_dataset/{wakeword}/{wakeword}_{voice_id}_{language}.wav"
                    # output_filename = "test2.wav"
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
    
    with open("EnglishNames.txt", 'r') as f:
        s = f.read()
    words = s.split('\n')
    for word in words:
        try:
            os.makedirs(f"./Audio_dataset/{word}", exist_ok=True)
        except Exception as e:
            print("Exception:", e)
    # print(words)
    # words = ["So, for so long, we have been depending on wakeword activated robots and assistants. When we presented Aira, our AI robot in Keraleeyam Fest, the audience didn't interact with it by saying the wakeword Aira, they just talked natural by asking questions to it and all. Thus Wakeword is unnatural. And its time that we changed that. Introducing SpeakSense, our first generation AI framework which replaces the Wakeword Technology."]
    for word in words:
        generate_wakewords(word)
        time.sleep(2)
    
    # for word in words:
    #     try:
    #         os.makedirs(f"./Audio_dataset/{word}", exist_ok=True)
    #     except Exception as e:
    #         print("Exception:", e)
    
    
# Shruti
# Mantra    
        
            
# Turn on the lights
# Turn off the lights
# Dim the lights
# Brighten the lights
# Set the lights to 50%
# Change light color to blue
# Open the door
# Close the door
# Lock the door
# Unlock the door
# Is the door open
# Play music
# Pause the music
# Next song
# Previous song
# Stop the music
# Increase the volume
# Mute the TV
# Turn on the TV
# What's the time?
# Set an alarm for 7 AM
# Set a reminder
# What's on my schedule
# What's the weather
# Will it rain today
# What’s the temperature
# Hey assistant
# Hello
# Can you hear me?
# Are you there?
# Increase fan speed
# Decrease fan speed
# What's the weather today
# Will it rain today
# What's the temperature outside
# How is the traffic
# Set an alarm for 7 AM
# Set a timer for 10 minutes
# Cancel the timer
# Remind me to drink water
# Add milk to my shopping list
# Remove eggs from my shopping list
# What’s on my calendar
# Add a meeting at 3 PM
# Call John
# Send a message to mom
# Read my notifications
# What time is it
# Tell me a joke
# What's the news today
# Start the vacuum
# Stop the vacuum
# Find my phone
# Where is my phone
# Switch on the heater
# Switch off the heater
# Start the coffee machine
# Stop the coffee machine
# What’s the battery level
# Turn on Bluetooth
# Turn off Bluetooth
# Enable Wi-Fi
# Disable Wi-Fi
# Connect to home network
# Disconnect from Wi-Fi
# Open the camera
# Take a selfie
# Record a video
# Stop recording
# Show me my gallery
# Open my photos
# Start a video call
# End the call
# Switch on airplane mode
# Turn off airplane mode
# What’s my location?
# Navigate to work
# How far is the airport
# Find the nearest restaurant
# Order pizza
# Order groceries
# Book a cab
# Schedule a ride for 8 PM
# Track my order
# Cancel my order
# Restart the device
# Shutdown the device
# Take a screenshot
# Open settings
# Enable dark mode
# Disable dark mode
# Open WhatsApp
# Send a voice message
# What's trending on Twitter