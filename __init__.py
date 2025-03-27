import easyocr
import requests
import numpy as np
from PIL import Image
from io import BytesIO


def perform_ocr_from_url(image_url, languages=['en']):
    """
    Perform OCR on an image from a URL using EasyOCR

    Args:
        image_url (str): URL of the image
        languages (list): List of language codes to use (default: English)

    Returns:
        List of detected text with their coordinates
    """
    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Open the image and convert to numpy array
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)

        # Create a reader object
        reader = easyocr.Reader(languages)

        # Read text from the image array
        results = reader.readtext(img_array)

        # Print the detected text
        print("Detected text:")
        for (bbox, text, prob) in results:
            print(f"- {text} (confidence: {prob:.2f})")

        return results

    except Exception as e:
        print(f"Error processing image: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Test with a sample image URL (replace with your URL)
    image_url = "https://cdn.zeptonow.com/production/ik-seo/tr:w-1000,ar-1000-1000,pr-true,f-auto,q-80/cms/product_variant/7b1c1a5c-a9bd-4353-8f5f-0209b1c29706/Leonardo-Extra-Light-Olive-Oil-Bottle.jpeg"

    detected_text = perform_ocr_from_url(image_url)

    # Optional: Save results to a text file
    if detected_text:
        with open("ocr_results", "w") as f:
            for (bbox, text, prob) in detected_text:
                f.write(f"{text}\n")
        print("Results saved to ocr_results.txt")