import csv

def get_item(image_path):
    from google.cloud import vision
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file("./fit-network-441921-b6-9a9b642aca0d.json")
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Print identified labels
    print("Ingredients identified:")
    for label in labels:
        print(label.description)
    
    return labels.description
def validate_recipes(image_folder, csv_path, output_csv="validation_results.csv"):
    """
    Validate recipe generation by comparing predictions with ground truth.

    Args:
        image_folder (str): Path to the folder containing images.
        csv_path (str): Path to the CSV file with ground truth labels.
            - First column: image file name.
            - Second column: expected output (ground truth).
        output_csv (str): Path to save the validation results.

    Returns:
        None
    """
    # Load ground truth labels from the CSV
    ground_truth = {}
    with open(csv_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            ground_truth[row[0]] = row[1]  # Map image file name to ground truth label

    # Prepare results storage
    results = []
    correct_predictions = 0
    total_predictions = len(ground_truth)

    print(f"Validating {total_predictions} images...")

    for image_file, expected_label in ground_truth.items():
        image_path = f"{image_folder}/{image_file}"

        # Call the `get_recipes` function to predict the recipe
        try:
            predictions = get_item(image_path)
            predicted_label = ", ".join(predictions)  # Combine predictions into a string

            # Check if the prediction matches the expected label
            is_correct = predicted_label.strip().lower() == expected_label.strip().lower()
            if is_correct:
                correct_predictions += 1

            # Store results
            results.append([image_file, expected_label, predicted_label, is_correct])

            print(f"Processed {image_file}: {'Correct' if is_correct else 'Incorrect'}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results.append([image_file, expected_label, "Error", False])

    # Save results to an output CSV file
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image File", "Expected Label", "Predicted Label", "Correct"])
        writer.writerows(results)

    # Summarize the results
    print(f"Validation complete.")
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {correct_predictions / total_predictions:.2%}")

if __name__ == "__main__":
    # Paths
    IMAGE_FOLDER = "./images"  # Path to folder containing images
    GROUND_TRUTH_CSV = "./ground_truth.csv"  # Path to CSV with ground truth labels
    OUTPUT_CSV = "./validation_results.csv"  # Path to save validation results

    # Run validation
    validate_recipes(IMAGE_FOLDER, GROUND_TRUTH_CSV, OUTPUT_CSV)
