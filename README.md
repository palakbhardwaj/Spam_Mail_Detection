# Spam_Mail_Detection
Project aims to classify emails as spam or not spam using machine learning techniques. The dataset is processed using `TfidfVectorizer` to convert text data into numerical feature vectors, and a Logistic Regression model is used to make predictions.
## Dataset

The dataset consists of text emails, with each email labeled as spam or not spam. It typically includes:
- `text`: The content of the email.
- `label`: The classification label indicating whether the email is spam (1) or not spam (0).

## Project Structure

- `data/`: Directory containing the dataset.
- `notebooks/`: Jupyter notebooks for data preprocessing, feature extraction, model training, and evaluation.
- `src/`: Source code for data processing, feature extraction, model training, and evaluation.
- `models/`: Directory to save trained models.
- `README.md`: Project documentation.

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/spam-mail-prediction.git
    cd spam-mail-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Preprocess the data and extract features using `TfidfVectorizer`:
    ```sh
    python src/preprocess_and_feature_extraction.py
    ```

2. Train the Logistic Regression model:
    ```sh
    python src/train_model.py
    ```

3. Evaluate the model:
    ```sh
    python src/evaluate_model.py
    ```

### Example

You can find a step-by-step example of data preprocessing, feature extraction, model training, and evaluation in the Jupyter notebook `Spam_Mail_Prediction_using_Machine_Learning.ipynb` located in the `notebooks/` directory.

## Data Analysis and Visualization

The project includes exploratory data analysis (EDA) to understand the distribution of spam and non-spam emails, visualize term frequencies, and evaluate model performance. Visualizations include histograms of term frequencies and performance metrics of the logistic regression model.

## Results

The Logistic Regression model is trained and evaluated using the TF-IDF features extracted from the email text. The model's performance metrics, such as accuracy, precision, recall, and F1-score, are provided in the evaluation section of the notebook.

