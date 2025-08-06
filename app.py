import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    print("Loading dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    X = X / 16.0  # scale pixel values to [0, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for ax in axes.ravel():
        idx = random.randrange(len(X_test))
        ax.imshow(X_test[idx].reshape(8, 8), cmap="gray")
        pred = clf.predict(X_test[idx].reshape(1, -1))[0]
        ax.set_title(f"True: {y_test[idx]}\nPred: {pred}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    print("Saved sample predictions to sample_predictions.png")


if __name__ == "__main__":
    main()
