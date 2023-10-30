import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
import time
import pickle
from PIL import Image



# Function to save the drawn image
def save_drawing_as_image(drawing, image_path):
    cv2.imwrite(image_path, drawing)

# Main Streamlit app
def main():
    st.title("Drawing to Image Saver")

    # Create a canvas for drawing with the provided settings
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="White",
        background_color="Black",
        height=400,
        key="canvas",
    )

    image_filename = "output_drawing.png"

    if st.button("Submit"):
        # Get the drawing from the canvas
        drawing = np.array(canvas_result.image_data)

        if drawing is not None:
            # Save the drawn image
            save_drawing_as_image(drawing, image_filename)

            # Display the saved image
            # st.image(image_filename, caption="Saved Drawing", use_column_width=True)

            with open('D:/Lab_Main/Sem_5/Capstone/mnistCNN/models_pickle/training_model5.pkl', 'rb') as file:
               model = pickle.load(file)
            
            image = Image.open('D:/Lab_Main/Sem_5/Capstone/mnistCNN/UI_Prediction/output_drawing.png')
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((28, 28))
            image = np.array(image) / 255.0  # Normalize the pixel values
            image = image.reshape(-1, 28, 28, 1)
            print(image)
            prediction = model[3].predict(image)

            st.subheader("Prediction")
            st.write(prediction)
    
    
    

if __name__ == "__main__":
    main()
