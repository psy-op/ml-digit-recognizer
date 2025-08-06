# MNIST Digit Recognizer

A simple machine learning app that trains a classifier to recognize handwritten digits using scikit-learn's digits dataset (a smaller version of MNIST).
The app uses Python and scikit-learn, and includes visualization of sample predictions.

## Features
- Loads and preprocesses the digits dataset (via `sklearn.datasets`)
- Trains a logistic regression classifier
- Shows accuracy on the test data
- Visualizes random predictions and saves them as an image

## Setup

1. **Clone the repo**
    ```bash
    git clone https://github.com/psy-op/ml-digit-recognizer.git
    cd ml-digit-recognizer
    ```

2. **Create a virtual environment (optional but recommended)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the app:
```bash
python app.py
```

## Output

- Test set accuracy printed in the console
- Sample predicted digits saved to `sample_predictions.png`

## Customization

- Try different classifiers (e.g., SVM, RandomForest)
- Save/load trained models
- Build a simple web UI with Flask or Streamlit (optional)

## License

MIT
