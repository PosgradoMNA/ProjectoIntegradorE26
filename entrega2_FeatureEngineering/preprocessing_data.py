from dataclasses import dataclass
import os
import cv2
import numpy as np


@dataclass
class PreprocessRawImages:
    raw_image_data_path: str
    cornea_label_data_path: str
    destination_path: str = "./extracted_cornea_area"

    def __post_init__(self):
        # Ensure destination folder exists
        if not os.path.exists(self.destination_path):
            os.makedirs(self.destination_path)

    def extract_cornea(self, eye_image, cornea_label):
        """
        Extracts the cornea area from the eye image using the cornea label (mask).
        Inverts the mask to ensure the black areas are kept.
        """
        if eye_image is None:
            raise ValueError("Error loading the eye image.")

        if cornea_label is None:
            raise ValueError("Error loading the cornea mask.")

        # Invert the mask so that black areas become white and vice versa
        inverted_mask = cv2.bitwise_not(cornea_label)

        # Apply the inverted mask on the eye image (keep only the area where the mask is black)
        extracted_cornea = cv2.bitwise_and(eye_image, eye_image, mask=inverted_mask)

        return extracted_cornea

    def smooth_reflective_area(self, extracted_cornea_image, threshold_value=180):
        """
        Detects and smooths out reflective or bright areas in the extracted cornea image.

        Parameters:
        - extracted_cornea_image: The input extracted cornea image (in BGR format).
        - threshold_value: The threshold for detecting bright areas. Pixels with values
          above this threshold will be considered reflective or bright (default 240).

        Returns:
        - Smoothed cornea image with reflective areas blended.
        """
        if extracted_cornea_image is None:
            raise ValueError("The extracted cornea image is not valid.")

        # Convert the cornea image to grayscale to detect bright spots
        gray_cornea_image = cv2.cvtColor(extracted_cornea_image, cv2.COLOR_BGR2GRAY)

        # Create a mask where bright pixels are detected
        _, bright_areas_mask = cv2.threshold(
            gray_cornea_image, threshold_value, 255, cv2.THRESH_BINARY
        )

        # Use inpainting to smooth out the bright areas
        smoothed_cornea = cv2.inpaint(
            extracted_cornea_image,
            bright_areas_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA,
        )

        return smoothed_cornea

    def process_eyes(self):
        """
        Processes each eye image and extracts the corneal area based on the corresponding mask.
        """
        # Iterate through all eye images
        for eye_image_name in os.listdir(self.raw_image_data_path):
            eye_image_path = os.path.join(self.raw_image_data_path, eye_image_name)

            # Get the corresponding mask image name and path
            cornea_label_path = os.path.join(
                self.cornea_label_data_path, eye_image_name.split(".")[-2] + ".png"
            )

            # Load the eye image and mask (mask should be grayscale)
            eye_image = cv2.imread(eye_image_path)
            if eye_image is None:
                print(f"Error loading eye image: {eye_image_path}")
                continue

            cornea_label = cv2.imread(cornea_label_path, cv2.IMREAD_GRAYSCALE)
            if cornea_label is None:
                print(f"Error loading mask: {cornea_label_path}")
                continue

            # Extract the cornea area
            extracted_cornea = self.extract_cornea(eye_image, cornea_label)

            # Smooth out reflective areas
            extracted_cornea = self.smooth_reflective_area(extracted_cornea)

            # Save the extracted cornea image
            output_path = os.path.join(self.destination_path, eye_image_name)
            cv2.imwrite(output_path, extracted_cornea)
            print(f"Saved extracted cornea to {output_path}")


if __name__ == "__main__":
    preprocess = PreprocessRawImages(
        raw_image_data_path="./data/rawImages",
        cornea_label_data_path="./data/corneaLabels",
    )

    preprocess.process_eyes()
