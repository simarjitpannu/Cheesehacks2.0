from google.cloud import vision
from google.oauth2 import service_account

# Authenticate using the service account JSON file
credentials = service_account.Credentials.from_service_account_file("./fit-network-441921-b6-9a9b642aca0d.json")
client = vision.ImageAnnotatorClient(credentials=credentials)

# Load your image file
with open("./ing images/milkeggs.jpg", "rb") as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Perform label detection
response = client.label_detection(image=image)
labels = response.label_annotations

# Print identified labels
print("Ingredients identified:")
for label in labels:
    print(label.description)

