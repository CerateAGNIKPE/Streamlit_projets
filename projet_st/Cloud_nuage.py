import streamlit as st
import pandas as pd
import numpy as np 
import os 
import random 

train_folder_path = 'train_images'
test_folder_path = 'test_images'

train = pd.read_csv("projet_st/train.csv")
train_copy = train.copy()
train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])

# Load train images
@st.cache_data
def Load_TrainImages(train_folder_path):
    train_image_files = os.listdir(train_folder_path)
    return  train_image_files

train_image_files =Load_TrainImages(train_folder_path)


def df_transform(df):
    
    # Remove rows with NaN in the 'EncodedPixels' column
    df = df.dropna(subset=['EncodedPixels'])
    
    # Perform one-hot encoding
    one_hot = pd.get_dummies(df['label'])
    # Concatenate the one-hot encoded columns to the original DataFrame
    df = pd.concat([df, one_hot], axis=1)
    
    # Group by 'image' and aggregate the values in the last four columns
    consolidated_df = df.groupby('image', as_index=False).agg({'Fish': 'max','Flower': 'max','Gravel': 'max','Sugar': 'max'})
    
    # Pivot the DataFrame to create separate columns for each label's 'EncodedPixels'
    pivot_df = df.pivot(index='image', columns='label', values='EncodedPixels')
    # Rename the columns with "_encoding"
    pivot_df.columns = [f'{label}_encoding' for label in pivot_df.columns]
    # Reset the index
    pivot_df = pivot_df.reset_index()
    
    merged_df = pd.merge(pivot_df, consolidated_df, on='image')
    return merged_df

# Transform Dataset for analysis usage
df_train = df_transform(train)

st.title("Projet of Understanding Cloud Organization")
st.sidebar.title("Table of contents")
pages = ['1_Projet background', '2_Dataset analysis', '3_Models-Our own built','4_Models-YoloV8','5_Performance comparaison']
page=st.sidebar.radio('Select chapter below', pages)

if page == pages[0]:
    
    st.markdown(
        """
        ### Understand and classify clouds from satellite images
        The topic of my project is [a Kaggle competition challenge](https://www.kaggle.com/competitions/understanding_cloud_organization/leaderboard), 
        but it was originally a research project, as described in the document [Combining crowd-sourcing and deep learning to explore the meso-scale 
        organization of shallow convection](https://arxiv.org/abs/1906.01906)
        
        ### What is the project goal?
        The original goal of the project is to combine crowd-sourcing and deep learning to explore the meso-scale organization of shallow convection. 
        It means using the collective efforts of many people (crowd-sourcing) and advanced computer algorithms (deep learning) to understand how 
        small-scale atmospheric processes called shallow convection are organized. Deep learning techniques, particularly computer vision, have shown promise in mimicking human pattern recognition abilities, 
        including in the analysis of satellite cloud imagery. 
        
        The challenge of the current project is to build a model to classify cloud organization patterns from satellite images.

        ### What is the target ouput?
        Object detection focuses on detecting and localizing specific objects or features of interest within an image. 
        The output of object detection algorithms is typically a set of bounding boxes that indicate the location and extent of the detected objects. 
        These bounding boxes will be associated with predefined 4-class labels.
        
        Semantic segmentation, on the other hand, aims to classify every pixel in an image according to the specific category or class it belongs to. 
        The output of semantic segmentation is the prediction and segmentation masks generated for each input image.
        """
    )

    st.image('target output.png')
if page == pages[1]:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    chapters = ["Load dataset", "Transform Dataset", "Explore images & labels", "Show some images & masks"]
    chapter = st.sidebar.radio("Select one section :", chapters)

    # page : Load dataset
    if chapter == chapters[0] : 
        st.write("The provided .csv file is as following")  
        st.dataframe(train_copy.head(8))
    # page : Transform Dataset
    elif chapter == chapters[1] : 
        st.write("We transformed the dataset into one image per row format") 
        st.dataframe(df_train.head()) 
    # Check unique images
        st.write("check if there any multipled image in dataset:") 
        st.write(f"Total number of images: {22184}")
        st.write(f"Number of unique image names: {5546}")
    elif chapter == chapters[2]:
        st.image('label.jpg')
        st.image('nb_label.jpg')

    # page : Show some images & masks
   
