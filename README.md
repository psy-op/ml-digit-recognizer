# MNIST Digit Recognizer

A simple machine learning app that trains a classifier to recognize handwritten digits using the classic MNIST dataset.  
The app uses Python and scikit-learn, and includes visualization of sample predictions.

## Features
- Loads and preprocesses the MNIST dataset (via `sklearn.datasets`)
- Trains a logistic regression classifier
- Shows accuracy on the test data
- Visualizes random predictions

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
- Plots of sample predicted digits

## Customization

- Try different classifiers (e.g., SVM, RandomForest)
- Save/load trained models
- Build a simple web UI with Flask or Streamlit (optional)

## License

MIT