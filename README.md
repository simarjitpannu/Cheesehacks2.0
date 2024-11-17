# Fridge2Fork

This project is a full-stack application designed to identify ingredients in a fridge from a live video feed, generate recipes based on the identified ingredients, and recommend those recipes to the user.

# Architecture Overview
The architecture consists of two main components: Frontend and Backend, each with distinct responsibilities that work together to achieve the desired functionality.

# 1. Frontend
The frontend is responsible for providing the user interface and capturing images from the live video feed.

Key Steps:
Live Video Feed:

Utilizes libraries like cv2 (OpenCV) and PyQt6 to access and display the live video feed from a webcam.
Convert Video to Image:

Captures frames from the video feed and saves them as images.
Pre-process the Image:

Performs necessary pre-processing (e.g., resizing, normalization) to prepare the image for backend processing.
Receive Recipe Recommendations:

Displays recommended recipes to the user after processing by the backend.
# 2. Backend
The backend handles ingredient identification and recipe generation using AI and external APIs.

Key Steps:
Ingredients Identification:

Processes the pre-processed image from the frontend.
Uses Google Vision API to identify the ingredients in the image.
Recipe Generation:

Passes the identified ingredients to a Large Language Model (LLM) (e.g., FlaxAutoModel) to generate personalized recipes based on the ingredients.
Send Recipes Back to Frontend:

Returns the generated recipes to the frontend for display and user interaction.

# Technologies Used
Frontend:

cv2 (OpenCV) for live video feed and image capture.
PyQt6 for building the graphical user interface.

Backend:

Google Vision API for image-to-ingredient processing.
FlaxAutoModel (or other LLM frameworks) for generating recipe recommendations.

# How It Works
Live Video Feed:

The frontend captures a live video feed of the fridge's contents.
Image Capture:

The user captures an image of the fridge's contents, which is pre-processed and sent to the backend.
Ingredient Detection:

The backend uses Google Vision API to analyze the image and identify the ingredients.
Recipe Generation:

The identified ingredients are passed to the LLM, which generates a list of recipes tailored to the available ingredients.
Recommendation:

The backend sends the generated recipes back to the frontend, where they are displayed for the user.
