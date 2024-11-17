import sys
import cv2
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
    QScrollArea,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt


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

        credentials = service_account.Credentials.from_service_account_file("./Cheesehacks2.0/fit-network-441921-b6-948aa79c1f80.json")
        client = vision.ImageAnnotatorClient(credentials=credentials)

        with open(path_to_image, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Perform label detection
        response = client.label_detection(image=image)
        labels = response.label_annotations

        ingredients = [
        "apple", "banana", "orange", "mango", "strawberry", "blueberry", "raspberry",
        "pineapple", "kiwi", "watermelon", "peach", "pear", "plum", "cherry", "lemon",
        "lime", "grapefruit", "fig", "pomegranate", "apricot", "avocado",
        "potato", "sweet potato", "carrot", "broccoli", "cauliflower", "spinach", "kale",
        "zucchini", "eggplant", "bell pepper (red, yellow, green)", "onion", "garlic",
        "ginger", "mushroom", "cucumber", "tomato", "corn", "lettuce", "cabbage",
        "celery", "asparagus", "green beans", "peas", "radish", "beetroot", "okra",
        "milk", "butter", "cream", "cheese (cheddar, mozzarella, parmesan, etc.)",
        "yogurt", "sour cream", "whipping cream", "cottage cheese",
        "chicken", "beef", "pork", "lamb", "turkey", "duck", "fish (salmon, tuna, cod, etc.)",
        "shrimp", "crab", "lobster", "scallops", "clams", "egg", "eggs"
        "chicken egg", "duck egg", "quail egg",
        "rice (white, brown, jasmine, basmati, etc.)", "quinoa", "oats", "barley", "couscous",
        "lentils (red, green, yellow)", "chickpeas", "black beans", "kidney beans",
        "pinto beans", "green peas",
        "salt", "black pepper", "paprika", "turmeric", "cinnamon", "nutmeg", "cloves",
        "basil", "oregano", "thyme", "rosemary", "parsley", "dill", "cilantro", "mint",
        "bay leaf", "chili powder", "curry powder",
        "olive oil", "vegetable oil", "canola oil", "coconut oil", "avocado oil",
        "butter", "ghee", "lard",
        "almonds", "cashews", "walnuts", "peanuts", "pistachios", "sunflower seeds",
        "chia seeds", "flaxseeds", "pumpkin seeds",
        "flour (all-purpose, whole wheat, almond, etc.)", "sugar (white, brown, powdered)",
        "baking powder", "baking soda", "yeast", "cocoa powder", "vanilla extract",
        "soy sauce", "tomato ketchup", "mayonnaise", "mustard", "vinegar (white, apple cider, balsamic)",
        "barbecue sauce", "hot sauce", "honey", "maple syrup",
        "bread", "tortillas", "pasta (spaghetti, penne, fettuccine, etc.)",
        "noodles (ramen, rice noodles, etc.)", "tofu", "tempeh", "coconut milk",
        "chocolate (dark, milk, white)", "jam or jelly", "pickles"]

        lab = []
        for label in labels:
            if label.description.lower() in ingredients:
                lab.append(label.description)

        return lab
    
    image_path_capture = #ENTER IMAGE PATH FOLDER
    items = get_ingredient_list(image_path_capture)
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
        self.setWindowTitle("Recipe Generator with Webcam Capture")
        self.setGeometry(100, 100, 900, 700)

        # Initialize OpenCV video capture
        self.cap = None
        self.current_camera_index = 0

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        # Top layout for dropdown and refresh button
        top_layout = QHBoxLayout()
        self.camera_dropdown = QComboBox()
        self.camera_dropdown.addItems(self.get_available_cameras())
        self.camera_dropdown.currentIndexChanged.connect(self.change_camera)
        top_layout.addWidget(self.camera_dropdown)

        self.refresh_button = QPushButton("Refresh Cameras")
        self.refresh_button.setFixedHeight(40)
        self.refresh_button.clicked.connect(self.refresh_cameras)
        top_layout.addWidget(self.refresh_button)
        self.layout.addLayout(top_layout)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setFixedSize(500, 300)  # Smaller camera feed
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Buttons layout
        button_layout = QHBoxLayout()

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.setFixedHeight(50)
        self.capture_button.clicked.connect(self.capture_image)
        button_layout.addWidget(self.capture_button)

        self.recipe_button = QPushButton("Generate Recipe")
        self.recipe_button.setFixedHeight(50)
        self.recipe_button.clicked.connect(self.generate_recipe)
        button_layout.addWidget(self.recipe_button)

        self.layout.addLayout(button_layout)

        # Scrollable area for recipe boxes
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.recipe_container = QWidget()
        self.recipe_layout = QVBoxLayout(self.recipe_container)
        self.recipe_layout.setContentsMargins(10, 10, 10, 10)
        self.recipe_layout.setSpacing(10)
        self.scroll_area.setWidget(self.recipe_container)
        self.layout.addWidget(self.scroll_area)

        # Timer for updating video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_feed)

        # Start the initial camera
        self.change_camera(0)

    def get_available_cameras(self):
        camera_names = []
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                camera_names.append(f"Camera {index}")
                cap.release()
        return camera_names if camera_names else ["No Cameras Available"]

    def refresh_cameras(self):
        self.camera_dropdown.clear()
        self.camera_dropdown.addItems(self.get_available_cameras())
        print("Camera list refreshed.")

    def change_camera(self, index):
        self.current_camera_index = index
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.current_camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.timer.start(30)
        else:
            self.timer.stop()
            self.video_label.clear()
            print(f"Failed to access Camera {self.current_camera_index}")

    def update_video_feed(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = rgb_image.shape
                q_image = QImage(rgb_image.data, width, height, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                # Scale the pixmap to fit the QLabel while maintaining aspect ratio
                self.video_label.setPixmap(
                    pixmap.scaled(
                        self.video_label.width(),
                        self.video_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

    def capture_image(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.captured_filename = (
                    f"captured_image_camera_{self.current_camera_index}_{timestamp}.jpg"
                )
                success = cv2.imwrite(self.captured_filename, frame)
                if success:
                    print(
                        f"Image captured and saved as {self.captured_filename}"
                    )
                else:
                    print("Failed to save the image.")

    def generate_recipe(self):
        try:
            recipes = get_recipes()
            # Clear previous recipes
            for i in reversed(range(self.recipe_layout.count())):
                widget = self.recipe_layout.takeAt(i).widget()
                if widget:
                    widget.deleteLater()

            # Display each recipe in its own box
            for recipe in recipes:
                recipe_box = QTextEdit()
                recipe_box.setText(recipe)
                recipe_box.setReadOnly(True)
                recipe_box.setFixedHeight(100)
                recipe_box.setStyleSheet(
                    "border: 1px solid #ccc; border-radius: 5px; padding: 5px; background-color: #000000;"
                )
                self.recipe_layout.addWidget(recipe_box)

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
