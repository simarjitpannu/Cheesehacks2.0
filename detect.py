import sys
import cv2
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QHBoxLayout,
    QTextEdit,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

# Placeholder for get_recipes function, which will be imported in your actual implementation
def get_recipes():
    from transformers import FlaxAutoModelForSeq2SeqLM
    from transformers import AutoTokenizer

    MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

    prefix = "items: "
    # generation_kwargs = {68
    #     "max_length": 512,
    #     "min_length": 64,
    #     "no_repeat_ngram_size": 3,
    #     "early_stopping": True,
    #     "num_beams": 5,
    #     "length_penalty": 1.5,
    # }
    generation_kwargs = {
        "max_length": 512,
        "min_length": 64,
    #  "no_repeat_ngram_size": 3,
        "do_sample": True,
        "top_k": 60,
        "top_p": 0.7
    }


    special_tokens = tokenizer.all_special_tokens
    tokens_map = {
        "<sep>": "--",
        "<section>": "\n"
    }
    def skip_special_tokens(text, special_tokens):
        for token in special_tokens:
            text = text.replace(token, "")

        return text

    def target_postprocessing(texts, special_tokens):
        if not isinstance(texts, list):
            texts = [texts]
        
        new_texts = []
        for text in texts:
            text = skip_special_tokens(text, special_tokens)

            for k, v in tokens_map.items():
                text = text.replace(k, v)

            new_texts.append(text)

        return new_texts

    def generation_function(texts):
        _inputs = texts if isinstance(texts, list) else [texts]
        inputs = [prefix + inp for inp in _inputs]
        inputs = tokenizer(
            inputs, 
            max_length=256, 
            padding="max_length", 
            truncation=True, 
            return_tensors="jax"
        )

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask


        output_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **generation_kwargs
        )
        generated = output_ids.sequences
        generated_recipe = target_postprocessing(
            tokenizer.batch_decode(generated, skip_special_tokens=False),
            special_tokens
        )
        return generated_recipe
    def get_ingredient_list(path_to_image):
        from google.cloud import vision
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file("./fit-network-441921-b6-9f9a9a0ee66c.json")
        client = vision.ImageAnnotatorClient(credentials=credentials)

        with open(path_to_image, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Perform label detection
        response = client.label_detection(image=image)
        labels = response.label_annotations

        # Print identified labels
        print("Ingredients identified:")
        lab = []
        for label in labels:
            lab.append(label.description)
        
        return lab
    
    image_path_capture = "./captured_image_camera_0_hq.jpg"
    items = get_ingredient_list(image_path_capture)
    print(items)
    items = ["chicken","beef","macaroni", "cheese"]
    generated = generation_function(items)
    final_string_ouput = ""
    for text in generated:
        sections = text.split("\n")
        for section in sections:
            section = section.strip()
            if section.startswith("title:"):
                section = section.replace("title:", "")
                headline = "TITLE"
            elif section.startswith("ingredients:"):
                section = section.replace("ingredients:", "")
                headline = "INGREDIENTS"
            elif section.startswith("directions:"):
                section = section.replace("directions:", "")
                headline = "DIRECTIONS"
            
            if headline == "TITLE":
                print(f"[{headline}]: {section.strip().capitalize()}")
            else:
                section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))]
                print(f"[{headline}]:")
                print("\n".join(section_info))

        print("-" * 130)
    return generated


class WebcamCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Webcam Capture App with Recipe Generator")
        self.setGeometry(100, 100, 800, 600)

        # Initialize OpenCV video capture
        self.cap = None  # Will be set later when a camera is selected
        self.current_camera_index = 0

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Top layout for dropdown and refresh button
        top_layout = QHBoxLayout()
        self.camera_dropdown = QComboBox()
        self.camera_dropdown.addItems(self.get_available_cameras())
        self.camera_dropdown.currentIndexChanged.connect(self.change_camera)
        top_layout.addWidget(self.camera_dropdown)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        top_layout.addWidget(self.refresh_button)
        self.layout.addLayout(top_layout)

        # Video display label
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        # Capture button
        self.capture_button = QPushButton("Capture High-Quality Image")
        self.capture_button.clicked.connect(self.capture_image)
        self.layout.addWidget(self.capture_button)

        # Generate Recipe button
        self.recipe_button = QPushButton("Generate Recipe")
        self.recipe_button.clicked.connect(self.generate_recipe)
        self.layout.addWidget(self.recipe_button)

        # Recipe display text area
        self.recipe_text = QTextEdit()
        self.recipe_text.setReadOnly(True)
        self.layout.addWidget(self.recipe_text)

        # Timer for updating video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_feed)

        # Start the initial camera
        self.change_camera(0)

    def get_available_cameras(self):
        """
        Detect available cameras on the system.
        Returns a list of camera names (e.g., 'Camera 0', 'Camera 1').
        """
        camera_names = []
        for index in range(10):  # Check up to 10 camera indices
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                camera_names.append(f"Camera {index}")
                cap.release()
        return camera_names if camera_names else ["No Cameras Available"]

    def refresh_cameras(self):
        """
        Refresh the list of available cameras and update the dropdown.
        """
        self.camera_dropdown.clear()
        self.camera_dropdown.addItems(self.get_available_cameras())
        print("Camera list refreshed.")

    def change_camera(self, index):
        """
        Switch to the selected camera and set resolution to the maximum supported resolution.
        """
        self.current_camera_index = index
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.current_camera_index)

        if self.cap.isOpened():
            # Set resolution to a high value (4K as an example; the camera will fallback if unsupported)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

            # Verify the resolution
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(
                f"Camera {self.current_camera_index} resolution set to: {width}x{height}"
            )

            # Resize the QLabel to match the camera's resolution
            self.video_label.setFixedSize(width, height)

            self.timer.start(30)  # 30 ms interval
        else:
            self.timer.stop()
            self.video_label.clear()
            print(f"Failed to access Camera {self.current_camera_index}")

    def update_video_feed(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB for display
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = rgb_image.shape
                q_image = QImage(
                    rgb_image.data, width, height, QImage.Format.Format_RGB888
                )
                pixmap = QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap)

    def capture_image(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Generate a unique filename using timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.captured_filename = (
                    f"captured_image_camera_{self.current_camera_index}_{timestamp}.jpg"
                )

                # Save the high-quality image with maximum JPEG quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]  # 100% JPEG quality
                success = cv2.imwrite(self.captured_filename, frame, encode_param)
                if success:
                    print(
                        f"High-quality image captured and saved as {self.captured_filename}"
                    )
                else:
                    print(f"Failed to save the image.")

    def generate_recipe(self):
        """
        Call the get_recipes function and display the output in the text area.
        """
        try:
            recipes = get_recipes()  # Replace this with the actual implementation
            self.recipe_text.clear()
            self.recipe_text.append("Generated Recipes:\n")
            for recipe in recipes:
                self.recipe_text.append(f"- {recipe}")
            print("Recipes displayed.")
        except Exception as e:
            print(f"Error generating recipes: {e}")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = WebcamCaptureApp()
    main_window.show()
    sys.exit(app.exec())
