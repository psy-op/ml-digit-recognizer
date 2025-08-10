import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def _synthesize_sequence(seq_str, images, y):
    # Build a horizontal concatenation of random examples for each digit in seq_str
    parts = []
    for ch in seq_str:
        d = int(ch)
        digit_indices = np.where(y == d)[0]
        idx = random.choice(digit_indices)
        parts.append(images[idx])
    return np.hstack(parts)  # shape (8, 8 * len(seq_str))


def _segment_sequence_image(img):
    # Assume contiguous fixed-width (8) digit blocks
    w = img.shape[1]
    n = w // 8
    return [img[:, i * 8:(i + 1) * 8] for i in range(n)]


def _predict_sequence(img, clf):
    segments = _segment_sequence_image(img)
    preds = []
    for seg in segments:
        pred = clf.predict(seg.reshape(1, -1))[0]
        preds.append(str(pred))
    return "".join(preds)


def main():
    print("Loading dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(0.95, random_state=42)),  # 95% variance retention
            ("logreg", LogisticRegression(max_iter=300, C=5, solver="lbfgs")),
        ]
    )
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

    # --- New multi-digit sequence demo ---
    sequences = ["120", "302", "928"]
    # Add one random 3-digit sample
    sequences.append("".join(str(random.randint(0, 9)) for _ in range(3)))

    seq_images = []
    for seq in sequences:
        img = _synthesize_sequence(seq, digits.images, y)
        pred_seq = _predict_sequence(img, clf)
        seq_images.append((seq, pred_seq, img))

    fig2, axes2 = plt.subplots(
        len(seq_images),
        1,
        figsize=(2 * max(len(s) for s, _, _ in seq_images), 2 * len(seq_images)),
    )
    if len(seq_images) == 1:
        axes_list = [axes2]
    else:
        axes_list = axes2
    for ax, (true_seq, pred_seq, img) in zip(axes_list, seq_images):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {true_seq}  Pred: {pred_seq}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sequence_predictions.png")
    print("Saved multi-digit sequence predictions to sequence_predictions.png")
    # --- End demo ---


if __name__ == "__main__":
    main()
