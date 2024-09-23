import cv2
import numpy as np
import pandas as pd
import pickle as pkl
import mahotas as mt
from PIL import Image
import streamlit as st
from catboost import CatBoostClassifier



def style():
    # Set page layout to wide
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
                background:  linear-gradient(90deg, hsla(236, 100%, 8%, 1) 0%, hsla(211, 100%, 28%, 1) 100%);
            }}


        </style>
        """,
        unsafe_allow_html=True
    )

    # Creating a title
    st.markdown("""<center class='custom-text'>🍁PLANT LEAF DISEASE DETECTION🍁</center>""", unsafe_allow_html=True)

def load_model(Plant):
    model = None

    with open(f"Models/{str(Plant).lower()}_model.pkl", 'rb') as file:
        model = pkl.load(file)

    return model

def image_uploader():
    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
    
        return image       

def image_preprocessing(pil_image):
    # Convert PIL image to a NumPy array
    image = np.array(pil_image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding
    otsu_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Invert the thresholded image
    invert = cv2.bitwise_not(otsu_image)

    # Dilate the image
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(invert, kernel, iterations=1)

    # Convert dilated image to 3-channel to match the original image
    dilated_image_3channel = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)

    # Bitwise AND operation between original image and dilated image
    result = cv2.bitwise_and(image, dilated_image_3channel)
    
    return result

def GLCM_And_HSV_Feature_Extraction(pil_image):
    # Preprocessed Image
    result = image_preprocessing(pil_image)

    # Convert the result to grayscale
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Calculate the co-occurrence matrix using mahotas
    glcm = mt.features.haralick(gray_result)

    # Calculate the mean of each feature across all directions (0°, 45°, 90°, and 135°)
    mean_glcm = glcm.mean(axis=0)

    # Convert the result image to HSV color space
    hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # Split the HSV image into individual H, S, and V channels
    h_channel, s_channel, v_channel = cv2.split(hsv_result)

    # Calculate mean and standard deviation for each HSV channel
    h_mean, h_std = np.mean(h_channel), np.std(h_channel)
    s_mean, s_std = np.mean(s_channel), np.std(s_channel)
    v_mean, v_std = np.mean(v_channel), np.std(v_channel)

    # Split the BGR image to get Red, Green, and Blue channels
    b_channel, g_channel, r_channel = cv2.split(result)

    # Calculate mean and standard deviation for each RGB channel
    red_mean, red_std = np.mean(r_channel), np.std(r_channel)
    green_mean, green_std = np.mean(g_channel), np.std(g_channel)
    blue_mean, blue_std = np.mean(b_channel), np.std(b_channel)

    # Threshold the grayscale image to extract contours for area and perimeter calculation
    _, thresholded = cv2.threshold(gray_result, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate area and perimeter (if contours are found)
    area = perimeter = 0
    if contours:
        # Assuming the largest contour represents the object
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

   
    # Define the names of the features we are interested in
    selected_feature_names = [
        'Contrast',
        'Dissimilarity',
        'Homogeneity (IDM)',
        'Energy',
        'Correlation',
        'Hue Mean',
        'Hue Std Dev',
        'Saturation Mean',
        'Saturation Std Dev',
        'Value Mean',
        'Value Std Dev',
        'Red Mean',
        'Green Mean',
        'Blue Mean',
        'Red Std Dev',
        'Green Std Dev',
        'Blue Std Dev',
        'Area',
        'Perimeter',
    ]

    # Select the corresponding indices for the features
    selected_features = [
        mean_glcm[1], mean_glcm[2], mean_glcm[4], mean_glcm[0], mean_glcm[2],
        h_mean, h_std, s_mean, s_std, v_mean, v_std,
        red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
        area, perimeter
    ]

    # Create a pandas DataFrame to hold the selected features
    df = pd.DataFrame([selected_features], columns=selected_feature_names)

    return df


style()

selection = st.selectbox("Select Plant Leaf:",
            ["Apple", "Peach", "Cherry", "Corn", "Grape", "Pepper", "Potato", "Tomato", "Strawberry"])

model = None

if selection == "Apple":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        


elif selection == "Peach":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        
  

elif selection == "Cherry":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        


elif selection == "Corn":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        
  

elif selection == "Grape":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        

elif selection == "Pepper":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        
   

elif selection == "Potato":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        


elif selection == "Tomato":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        


elif selection == "Strawberry":
    if selection:
        model = load_model(selection)
        if model:
            image = image_uploader()
            if image is not None:
                df = GLCM_And_HSV_Feature_Extraction(image)
                c1, c2, c3, c4 = st.columns(4)
                with c2:
                    st.image(image, caption="Leaf Image", use_column_width=False, width=200)
                with c3:
                    st.markdown(f"""<h2><br>{str(model.predict(df)[0]).replace('[','').replace("]",'').replace("'",'')}</h2>""", unsafe_allow_html=True)        
   


