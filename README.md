# Potato-Leaf-Disease-Detection

# 🍃 Potato Leaf Disease Detection  
A Deep Learning-based model to classify potato leaves as **Healthy, Early Blight, or Late Blight** using **Convolutional Neural Networks (CNNs)**. The model is deployed as a **Streamlit web application** for real-time disease detection.  

## 🚀 Project Overview  
This project aims to assist farmers in **early disease detection** to improve crop yield and reduce losses. It follows a structured pipeline from **data acquisition to model deployment**.  

## 🛠️ Tech Stack  
- **TensorFlow/Keras** → Model Training  & Image Processing  
- **Streamlit** → Web App Deployment  
- **NumPy & Pandas** → Data Handling  
- **Matplotlib & Seaborn** → Visualization  

## 📌 Workflow  
1. **Data Collection** → Gathered a dataset of potato leaf images (Healthy, Early Blight, Late Blight).  
2. **Preprocessing** → Resized images (128x128), normalized pixel values, and applied augmentation.  
3. **Feature Extraction** → Used a **CNN model** to extract patterns from images.  
4. **Model Training** → Trained a CNN with **Categorical Cross-Entropy loss** and **Adam optimizer**.  
5. **Evaluation** → Achieved **high accuracy** using validation datasets.  
6. **Deployment** → Integrated the model into a **Streamlit-based web app** for user predictions.  

