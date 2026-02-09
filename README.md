# Image Classification Using CNN on CIFAR‑10
### Project Summary
This project builds an image classifier using a Convolutional Neural Network (CNN) trained on the CIFAR‑10 dataset.

The goal is to classify small 32×32 color images into one of ten categories:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck
  
The project covers the full deep‑learning workflow: loading the dataset, preparing the images, building a CNN model, training it, evaluating performance, and saving the final model.

An optional Gradio interface is also included to make predictions on uploaded images.

### Dataset Overview
CIFAR‑10 is a widely used dataset for image classification. It contains:
- 50,000 training images
- 10,000 test images
- All images are 32×32 pixels, in color
- Ten balanced classes
  
Before training, all images are normalized so the model can learn more effectively.

A preview of sample images is displayed to understand the dataset visually.

### Step 1: Environment Setup
The required libraries are installed, including TensorFlow, Matplotlib, scikit‑learn, and Gradio.

Random seeds are set to ensure consistent results each time the notebook is run.

### Step 2: Load CIFAR‑10 Dataset
The dataset is loaded into training and test sets.

Each image is paired with a label representing one of the ten classes.

The images are converted to floating‑point values between 0 and 1.

### Step 3: Dataset Preview
A small grid of sample images is shown along with their class names.

This helps confirm that the dataset loaded correctly and gives a quick visual understanding of the image types.

### Step 4: Hyperparameters
The project defines key settings such as:
- Image size
- Batch size
- Number of epochs
- Number of classes
  
These values control how the model trains.

### Step 5: Data Generators (Augmentation)
Data augmentation is applied to the training images to help the model generalize better.

This includes:
- Small rotations
- Horizontal shifts
- Vertical shifts
- Horizontal flips
  
The test set is not augmented.

Both training and test sets are converted into data generators for efficient loading during training.

### Step 6: Build CNN Model
A CNN model is created using multiple layers:
- Convolution layers to extract features
- MaxPooling layers to reduce spatial size
- Dropout layers to prevent overfitting
- A dense layer for learning patterns
- A final softmax layer for class prediction
  
The model is designed to be simple, fast, and effective for CIFAR‑10.

### Step 7: Compile Model
The model is compiled using:
- Adam optimizer
- Categorical cross‑entropy loss
- Accuracy as the evaluation metric
  
This prepares the model for training.

### Step 8: Train the Model
To speed up training, a smaller subset of 10,000 images is used.

The model is trained for several epochs using the augmented training data.

Validation is performed on the full test set to monitor performance.

### Step 9: Plot Accuracy & Loss
Two graphs are generated:
- Training vs validation accuracy
- Training vs validation loss
  
These plots help visualize how well the model is learning and whether it is overfitting or underfitting.

### Step 10: Evaluation
The model is evaluated using:
- A classification report
    - Precision
    - Recall
    - F1‑score
- A confusion matrix
    - Shows correct and incorrect predictions
    - Helps identify which classes are more challenging
  
This gives a clear picture of the model’s strengths and weaknesses.

### Step 11: Save Model
The trained model is saved as a .h5 file so it can be reused later without retraining.

### Step 12: Optional Gradio App
A simple Gradio interface is created to allow users to:
- Upload an image
- Automatically resize it
- Run it through the model
- View the predicted class probabilities
  
This makes the project interactive and easy to demonstrate.

### Final Conclusion
Overall Findings
- The CNN model performs well on CIFAR‑10 even when trained on a smaller subset.
- Data augmentation improves generalization and reduces overfitting.
- Accuracy and loss curves show steady learning across epochs.
- The classification report and confusion matrix provide detailed insights into performance.
- The saved model and Gradio interface make the project easy to reuse and present.
  
### Summary
This project demonstrates a complete deep‑learning workflow for image classification, including:
- Data loading
- Preprocessing
- Augmentation
- Model building
- Training
- Evaluation
- Visualization
- Deployment with Gradio
