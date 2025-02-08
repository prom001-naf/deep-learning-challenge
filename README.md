# Deep-Learning-Challenge

## Overview

This repository provides step-by-step instructions for preprocessing data in preparation for training and evaluating a neural network model. The dataset, `charity_data.csv`, will be cleaned, transformed, and scaled using Pandas and scikit-learn libraries.

---

## Prerequisites

Before starting, ensure you have the following installed:

- Python 3.8+
- Pandas
- scikit-learn
- Google Colab (optional but recommended)

---

## Instructions

### Step 1: Preprocess the Data

1. **Load the Dataset**  
   - Upload the starter file to Google Colab.
   - Use the provided cloud URL to read `charity_data.csv` into a Pandas DataFrame.

2. **Identify Variables**  
   - Determine the **target variable(s)** and **feature variable(s)** for your model.

3. **Drop Irrelevant Columns**  
   - Remove the `EIN` and `NAME` columns from the DataFrame.

4. **Analyze Column Values**  
   - Count the number of unique values in each column.
   - For columns with more than 10 unique values:
     - Calculate the number of data points for each unique value.
     - Combine "rare" categorical variables into a new category called `Other`.

5. **Encode Categorical Variables**  
   - Use `pd.get_dummies()` to perform one-hot encoding of the categorical variables.

6. **Split the Data**  
   - Split the preprocessed data into:
     - Features array: `X`
     - Target array: `y`
   - Use `train_test_split` to divide the data into training and testing sets.

7. **Scale the Data**  
   - Create a `StandardScaler` instance.
   - Fit the scaler to the training data and transform both the training and testing datasets.

## Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. **Design the Model**  
   - Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
   - Create the first hidden layer and choose an appropriate activation function.
   - If necessary, add a second hidden layer with an appropriate activation function.
   - Create an output layer with an appropriate activation function.
   - Check the structure of the model.

3. **Compile and Train the Model**  
   - Compile the model using a suitable loss function, optimizer, and evaluation metric(s).
   - Train the model using the training data.

4. **Add Callback for Saving Weights**  
   - Create a callback that saves the model's weights every five epochs.

5. **Evaluate the Model**  
   - Evaluate the model using the test data to determine the loss and accuracy.

6. **Save the Model**  
   - Save and export the trained model to an HDF5 file named `AlphabetSoupCharity.h5`.

## Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

### Optimization Methods
Use any or all of the following techniques to improve the model's performance:

- **Input Data Adjustments**:
  - Drop additional or fewer columns.
  - Create more bins for rare occurrences in categorical columns.
  - Adjust the number of values for each bin (increase or decrease).

- **Model Adjustments**:
  - Add more neurons to a hidden layer.
  - Add additional hidden layers.
  - Experiment with different activation functions for the hidden layers.
  - Modify the number of epochs during training (increase or decrease).

**Note:** If you make at least three optimization attempts, you will not lose points even if the model does not achieve the target performance.

---

### Instructions

1. **Set Up the Optimization Environment**  
   - Create a new Google Colab file and name it `AlphabetSoupCharity_Optimization.ipynb`.
   - Import all required dependencies.

2. **Load the Dataset**  
   - Read in the `charity_data.csv` file to a Pandas DataFrame using the provided cloud URL.

3. **Data Preprocessing**  
   - Preprocess the dataset as performed in Step 1, incorporating any adjustments that arise from the optimization process.

4. **Design the Optimized Neural Network Model**  
   - Adjust the model's architecture to optimize for an accuracy above 75%. This may include:
     - Increasing or decreasing the number of neurons.
     - Adding or removing hidden layers.
     - Trying different activation functions.

5. **Train and Save the Optimized Model**  
   - Train the optimized model.
   - Save the final model to an HDF5 file named `AlphabetSoupCharity_Optimization.h5`.

6. **Export Results**  
   - Save and export your results for evaluation.

## Step 4: Write a Report on the Neural Network Model

For this part of the assignment, write a comprehensive report analyzing the performance of the deep learning model created for Alphabet Soup. The report should cover the following sections:

---

### Overview of the Analysis

Provide a summary of the purpose of this analysis. Explain why the deep learning model was created and the goals of the classification task.

---

### Results

Use bulleted lists and visual aids (e.g., images, graphs, or tables) to support your analysis and address the following questions:

#### Data Preprocessing
- **Target Variable(s)**: Identify the variable(s) that serve as the target(s) for your model.
- **Feature Variable(s)**: Identify the variable(s) used as input features for your model.
- **Irrelevant Variables**: Specify the variables removed from the input data because they are neither targets nor features.

#### Compiling, Training, and Evaluating the Model
- **Neural Network Architecture**:
  - How many neurons, layers, and activation functions did you select for your model, and why?
- **Performance**:
  - Did your model achieve the target accuracy of 75%? Provide a summary of the final accuracy and loss.
- **Optimization Attempts**:
  - What steps did you take to improve model performance? Include details of any changes to the data, architecture, or training process.

---

### Summary

Summarize the overall results and insights gained from the analysis:
- Highlight the key outcomes of the deep learning model.
- Provide a recommendation for how a different model (e.g., decision trees, random forest, or ensemble methods) might solve the classification problem.
- Justify your recommendation based on the strengths and weaknesses of the current model.

---

Include your report in a markdown cell or as a separate Markdown/HTML file for submission.

---
