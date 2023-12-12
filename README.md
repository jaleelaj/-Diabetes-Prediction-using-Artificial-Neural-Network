# -Diabetes-Prediction-using-Artificial-Neural-Network
Overview
This project aims to predict diabetes using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The ANN is trained on a dataset containing various features related to diabetes and utilizes a binary classification approach to predict the presence or absence of diabetes.

Project Structure
The project consists of the following main components:

Data Processing:

Loading the dataset from 'diabetes.csv'.
Preprocessing steps including feature scaling and splitting data into training and testing sets.
Building the ANN:

Creating a Sequential model in Keras.
Adding Dense layers with ReLU activation functions.
Compiling the model with binary cross-entropy loss and accuracy metrics.
Training the Model:

Fitting the model on the training data with validation split and specified epochs and batch size.
Visualizing the model's training history with accuracy and loss plots.
Evaluation:

Making predictions on the test set.
Generating a confusion matrix to evaluate model performance.
Calculating the accuracy score.
Instructions
Requirements
TensorFlow
NumPy
Matplotlib
Pandas
Scikit-learn
How to Run
Ensure all dependencies are installed.
Clone the repository and navigate to the project directory.
Place the 'diabetes.csv' dataset in the project folder.
Run the code in the specified sequence as provided in the script.
Results
The accuracy of the trained model on the test set is calculated to be 79.87. The confusion matrix provides insights into the model's performance in predicting diabetes.
