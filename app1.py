# Import libraries
import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os
import shutil

# loading model and image processing
checkpoint = "FatihC/swin-tiny-patch4-window7-224-finetuned-eurosat-watermark"
model = AutoModelForImageClassification.from_pretrained(checkpoint)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# defining class and output folder
class_names = ["non-watermarked", "watermarked"]
output_folders = ["non-watermarked", "watermarked"]

# Create the Streamlit app
def main():
    st.title("Image Watermark Classification")

    # Allow user to select input folder
    input_folder = st.file_uploader("Select input folder:", type='folder')

    if input_folder:
        # Create output folders if they don't exist
        for folder in output_folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Loop through the input folder
        for filename in os.listdir(input_folder):
            # Load and preprocess the image
            image = Image.open(os.path.join(input_folder, filename))
            inputs = image_processor(image, return_tensors="pt")

            # Make a prediction
            outputs = model(**inputs)
            output001 = outputs[0][0][1]
            output000 = outputs[0][0][0]

            if output000 > 0.20 and output001 < -0.1:
                preds = 0
            else:
                preds = outputs.logits.argmax(-1).item()

            # Get the predicted label and the corresponding output folder
            label = class_names[preds]
            output_folder = output_folders[preds]

            # Copy the image to the output folder
            shutil.copy(os.path.join(input_folder, filename), output_folder)

            # Display the filename and the predicted label
        st.write(f"Task completed successfully")

if __name__ == "__main__":
    main()
