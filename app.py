import cv2
import numpy as np
import pandas as pd
import pickle as pkl
import mahotas as mt
from PIL import Image
import streamlit as st
from catboost import CatBoostClassifier

def style():
    """Set up the Streamlit app's layout and custom styling."""
    # Set the page layout to wide
    st.set_page_config(layout="wide")

    # Inject custom CSS for page styling
    st.markdown(
        f"""
        <style>
            /* Background color and image */
            .stApp {{
                background-image: url(https://img.freepik.com/premium-photo/silhouetted-blue-leaves-pink-teal-gradient-background_1306868-12515.jpg?w=1060);
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                text-color: black;
            }}

            .main {{
                overflow-x: hidden;
            }}

            #MainMenu {{visibility: hidden;}}
            header {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            
            #root > div:nth-child(1) > div > div > div > div > section > div {{
                padding-top: 1rem; 
                padding-left: 2rem;
                padding-right: 2rem; 
                margin: 1rem;
            }}
            
            .custom-title {{
                font-family: 'Arial', sans-serif;
                text-align: center;
                margin-bottom: 18px;
                letter-spacing: 1px;
                text-transform: uppercase;
                font-size: 45px;
                font-weight: 800;
                color:white;
            }}

            .custom-text {{
                @import url('https://fonts.googleapis.com/css2?family=Akaya+Telivigala&display=swap'); 
                font-family: 'Akaya Telivigala', cursive;
                font-size: 35px;
                font-weight: bold;
                text-align: center;
                color: white;
                text-shadow: 2px 2px 5px rgba(1, 1, 1, 0.4);
            }}

            .st-emotion-cache-1erivf3{{
                color:white;
                font-size:16px;
                font-weight:bold;
                background: linear-gradient(90deg, hsla(236, 100%, 8%, 1) 0%, hsla(211, 100%, 28%, 1) 100%);
            }}

            .st-emotion-cache-19cfm8f{{                
                color:white;
                font-size:16px;
                font-weight:bold;
            }}

            .st-gj, .st-bb{{
                color: white;
                font-size: 20px;
                font-weight:bold;
                content-align:center;
                width:auto;
                background: linear-gradient(90deg, hsla(236, 100%, 8%, 1) 0%, hsla(211, 100%, 28%, 1) 100%);
            }}

            .stButton > button {{
                width: 200px;  /* Set your desired width */
                height: 60px;  /* Optionally set a height */
                font-size: 22px;  /* Adjust font size if needed */
                padding:10px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Creating a title for the app
    st.markdown("""<center class='custom-text'>🍁PLANT LEAF DISEASE DETECTION🍁</center>""", unsafe_allow_html=True)

def load_model(plant):
    """Load the pre-trained model for the specified plant type.

    Args:
        plant (str): The type of plant for which to load the model.

    Returns:
        model: The loaded machine learning model.
    """
    model = None
    with open(f"Models/{str(plant).lower()}_model.pkl", 'rb') as file:
        model = pkl.load(file)
    return model

def image_uploader():
    """Create an image uploader for users to upload leaf images.

    Returns:
        image: The uploaded image in PIL format or None if no image is uploaded.
    """
    # Upload image
    uploaded_image = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"], key='image_uploader')
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        return image       
    return None

def image_preprocessing(pil_image):
    """Preprocess the uploaded image for feature extraction.

    Args:
        pil_image: The uploaded image in PIL format.

    Returns:
        result: The processed image ready for feature extraction.
    """
    # Convert PIL image to NumPy array
    image = np.array(pil_image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Otsu's thresholding
    otsu_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Invert the image
    invert = cv2.bitwise_not(otsu_image)

    # Dilate the image
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(invert, kernel, iterations=1)

    # Convert back to 3 channels
    dilated_image_3channel = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)

    # Apply bitwise AND to get the result
    result = cv2.bitwise_and(image, dilated_image_3channel)
    return result

def GLCM_And_HSV_Feature_Extraction(pil_image):
    """Extract features from the preprocessed image using GLCM and HSV.

    Args:
        pil_image: The preprocessed image in PIL format.

    Returns:
        df: A DataFrame containing the extracted features.
    """
    result = image_preprocessing(pil_image)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Extract GLCM features
    glcm = mt.features.haralick(gray_result)
    mean_glcm = glcm.mean(axis=0)
    
    # Convert image to HSV color space
    hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_result)

    # Calculate HSV channel statistics
    h_mean, h_std = np.mean(h_channel), np.std(h_channel)
    s_mean, s_std = np.mean(s_channel), np.std(s_channel)
    v_mean, v_std = np.mean(v_channel), np.std(v_channel)

    # Split the BGR channels and calculate their statistics
    b_channel, g_channel, r_channel = cv2.split(result)
    red_mean, red_std = np.mean(r_channel), np.std(r_channel)
    green_mean, green_std = np.mean(g_channel), np.std(g_channel)
    blue_mean, blue_std = np.mean(b_channel), np.std(b_channel)

    # Find contours and calculate area and perimeter
    _, thresholded = cv2.threshold(gray_result, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

    # Define selected feature names
    selected_feature_names = [
        'Contrast', 'Dissimilarity', 'Homogeneity (IDM)', 'Energy',
        'Correlation', 'Hue Mean', 'Hue Std Dev', 'Saturation Mean',
        'Saturation Std Dev', 'Value Mean', 'Value Std Dev',
        'Red Mean', 'Green Mean', 'Blue Mean', 'Red Std Dev',
        'Green Std Dev', 'Blue Std Dev', 'Area', 'Perimeter',
    ]

    # Create a list of selected features
    selected_features = [
        mean_glcm[1], mean_glcm[2], mean_glcm[4], mean_glcm[0], mean_glcm[2],
        h_mean, h_std, s_mean, s_std, v_mean, v_std,
        red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
        area, perimeter
    ]

    # Create a DataFrame to hold the features
    df = pd.DataFrame([selected_features], columns=selected_feature_names)
    return df

# Apply styling to the Streamlit app
style()

# Dropdown for selecting plant type
selection = st.selectbox("Select Plant Type:", ["Apple", "Peach", "Cherry", "Corn", "Grape", "Pepper", "Potato", "Tomato", "Strawberry"])

# Create columns for image upload and prediction button
c1, c2 = st.columns([4, 1])
with c1:
    uploaded_image = image_uploader()  # Call the image uploader function
with c2:
    st.title("")  # Placeholder title
    btn = st.button("Predict")  # Prediction button

# Action when the prediction button is clicked
if btn:
    model = load_model(selection)  # Load the model for the selected plant type
    if uploaded_image is not None:  # Ensure an image was uploaded
        df = GLCM_And_HSV_Feature_Extraction(uploaded_image)  # Extract features from the image
        
        # Create columns for displaying the image and prediction results
        c1, c2, c3, c4 = st.columns(4)
        with c2:
            st.image(uploaded_image, caption="Leaf Image", use_column_width=False, width=200)  # Display the uploaded image
            
        prediction = model.predict(df)[0]  # Make a prediction using the model
        prediction_result = str(prediction).replace('[', '').replace("]", '').replace("'", '')
        result = " ".join(str(prediction_result).split("___")[1:])  # Format the prediction result
        result = " ".join(str(result).split("_"))  # Clean up the result string
        
        # Display the prediction result
        with c3:
            st.markdown(f"""<h2 style="@import url('https://fonts.googleapis.com/css2?family=Akaya+Telivigala&display=swap'); 
                            font-family: 'Akaya Telivigala', cursive;
                            font-size: 24px;
                            text-align: center;
                            text-shadow: 2px 2px 5px rgba(1, 1, 1, 0.4);"><br>{result}</h2>""", unsafe_allow_html=True)
