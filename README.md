# EyeDiseasesClassification_WEB
# Eye Disease Classification using Retinal Fundus Images

This project focuses on the classification of eye diseases using retinal fundus images. The model is based on a convolutional neural network (CNN), specifically **EfficientNetB3**, which is well-suited for medical image classification due to its efficiency and accuracy. The model was retrained on a custom dataset with some modifications to the original code, and a simple web interface was built using Flask to allow users to upload or drag-and-drop eye images for disease prediction.

## Project Overview

The goal of this project is to classify eye diseases from retinal fundus images. The model was initially developed by another [developer](https://www.kaggle.com/code/abdallahwagih/eye-diseases-classification-acc-93-8), and I retrained it with a new dataset, made some modifications, and deployed it using a Flask web interface. The model achieves an accuracy of **0.95** on the testing dataset, but its performance in real-world applications may vary depending on factors like image quality, illumination, and data augmentation.

Here’s a concise section you can add to your `README.md` to showcase the images of your project:

---

## Project Demo

Here are some screenshots of the web interface and its functionality:

1. **Main Interface**  
   ![Main Interface](https://github.com/JuanHoKKeR/EyeDiseasesClassification_WEB/blob/main/images/Webinterface.png)  
   *The main page of the web interface where users can upload or drag-and-drop retinal fundus images.*

2. **Drag and Drop Example**  
   ![Drag and Drop](https://github.com/JuanHoKKeR/EyeDiseasesClassification_WEB/blob/main/images/draganddrop.png)
   *An example of dragging and dropping an image into the interface.*

3. **Image Loaded**  
   ![Image Loaded](https://github.com/JuanHoKKeR/EyeDiseasesClassification_WEB/blob/main/images/beforeToPredict.png)  
   *The interface after an image has been successfully uploaded.*

4. **Prediction Result**  
   ![Prediction Result](https://github.com/JuanHoKKeR/EyeDiseasesClassification_WEB/blob/main/images/diseasePredict.png)  
   *The model's prediction result displayed on the interface.*

---

### Key Features
- **Retinal Fundus Image Analysis**: The model is trained on retinal fundus images to classify eye diseases.
- **EfficientNetB3**: The convolutional neural network used for training, known for its effectiveness in medical image classification.
- **Data Augmentation**: Techniques were applied to improve the model's robustness.
- **Flask Web Interface**: A simple web interface allows users to upload or drag-and-drop images for prediction.
- **Custom Callbacks and Visualization**: The original code includes personalized callbacks, training history plots, and confusion matrix visualization.

---

## Model Details

### Dataset
The model was trained on a dataset of retinal fundus images of Kaggle. The dataset includes images of various eye diseases, and the model was retrained to improve its performance on this specific data.

### Model Architecture
The model uses **EfficientNetB3** as the backbone for feature extraction. This architecture is particularly effective for medical image classification due to its ability to handle complex patterns in images with high accuracy.

### Training and Evaluation
- **Training**: The model was retrained with a custom dataset, and data augmentation techniques were applied to improve generalization.
- **Evaluation**: The model achieved an accuracy of **0.95** on the testing dataset. However, real-world performance may vary due to differences in image quality and illumination.

### Limitations
- The model may struggle with images that have significantly different illumination or quality compared to the training data.
- Further improvements can be made by incorporating more diverse data and advanced augmentation techniques.

---

## Web Interface

A simple web interface was built using **Flask** to allow users to interact with the model. Users can upload or drag-and-drop retinal fundus images, and the model will predict the corresponding eye disease.

### How to Use the Web Interface
1. Launch the Flask application.
2. Open the web interface in your browser.
3. Upload or drag-and-drop a retinal fundus image.
4. The model will process the image and display the predicted disease.

---

## Acknowledgments

This project is based on the original work by **[Abdallah Wagih Ibrahim](https://www.kaggle.com/code/abdallahwagih/eye-diseases-classification-acc-93-8)**. The original code includes functions for:
- Creating a dataframe from the dataset.
- Data augmentation.
- Displaying data samples.
- Personalized callbacks during training.
- Plotting training history and confusion matrix.

I retrained the model with a new dataset, made modifications, and built the Flask web interface for deployment.

---

## Installation and Usage

### Prerequisites
- Python 3.10
- TensorFlow 2.9
- Flask
- Other required libraries (requirements.txt)


## Future Work
- Incorporate more diverse datasets to improve the model's robustness.
- Experiment with advanced data augmentation techniques to handle variations in image quality and illumination.
- Deploy the model on a cloud platform for wider accessibility.

---
