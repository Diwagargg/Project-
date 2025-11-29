Network Intrusion Detection Using Neural Networks & Clustering

This project implements a Machine Learning–based Intrusion Detection System (IDS) using the KDD99 dataset.
It performs data preprocessing, neural-network classification using TensorFlow, clustering using KMeans, and several visualizations to analyze network traffic and attack patterns.

📌 Features

✔ Load and preprocess KDD99 network traffic data
✔ Convert categorical features to numeric using Label Encoding
✔ Normalize features with StandardScaler
✔ Train/Test split for supervised learning
✔ Build and train a Neural Network classifier using TensorFlow
✔ Evaluate model performance (Accuracy, Precision, Recall, F1, ROC-AUC)
✔ Save trained model (tf_model.keras)
✔ Apply K-Means clustering to find hidden patterns
✔ Visualize:

Confusion Matrix

ROC Curve

Label distribution

Scatter plot of src_bytes vs dst_bytes

Distribution of connection duration

📂 Dataset

This project uses kdd_test.csv, a portion of the KDD Cup 1999 Intrusion Detection dataset.

Column names are manually assigned based on the dataset structure.

📦 Dependencies

Install required packages:

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow


When using Google Colab, the dataset is uploaded via:

from google.colab import files
data = files.upload()

🚀 How the Code Works
1. Load and Clean Data

Remove duplicates

Convert features to numeric

Replace missing values with median

2. Encode Categorical Features

Features like protocol_type, service, flag are label-encoded.

3. Normalize Data

StandardScaler is used for consistent scaling.

4. Train Neural Network

A simple neural network:

Dense(32 → 16 → 1)
Activation: ReLU + Sigmoid
Loss: Binary Crossentropy

5. Evaluate Model

Prints:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

6. Visualizations

Includes:

Confusion Matrix

ROC Curve

Countplot of labels

Scatter plot for traffic bytes

Duration distribution

7. Clustering

KMeans (2 clusters) is applied to reveal structure in data.

📊 Model Output Files

The trained model is saved as:

tf_model.keras


You can load it later using:

model = tf.keras.models.load_model("tf_model.keras")

📈 Example Visualizations

The script generates:

Confusion Matrix

ROC Curve

Scatter Plot

Histograms and KDE distributions

These help understand dataset characteristics and classification effectiveness.

🧠 Future Improvements

Add more complex deep learning models (CNN, LSTM)

Test on full KDD99 or NSL-KDD dataset

Apply SMOTE to handle class imbalance

Feature selection (PCA, chi-square, mutual information)

📜 License

This project is open-source and free to use for research and education.
