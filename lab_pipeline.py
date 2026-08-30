import os
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. FASTA LOADING
# ============================================================

def load_fasta(file_path):
    sequences = []

    with open(file_path, "r") as f:
        seq = ""

        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.upper()

        if seq:
            sequences.append(seq)

    return sequences


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATA_DIR = "data"

sequences = []
labels = []

for filename in sorted(os.listdir(DATA_DIR)):

    if filename.endswith(".fasta"):

        species_name = filename.replace(".fasta", "")

        # ZADATAK 2:
        # Remove frog_species_3
        if species_name == "frog_species_3":
            continue

        file_path = os.path.join(DATA_DIR, filename)

        seqs = load_fasta(file_path)

        for seq in seqs:

            # Divide each sequence into smaller fragments
            fragment_length = 100

            for start in range(
                0,
                len(seq) - fragment_length + 1,
                fragment_length
            ):

                fragment = seq[
                    start:start + fragment_length
                ]

                sequences.append(fragment)
                labels.append(species_name)

        print(
            f"{species_name}: "
            f"{len(seqs)} original sequence(s)"
        )


print()
print(f"Loaded {len(sequences)} fragments")
print(f"Number of species: {len(set(labels))}")

print("\nFragments per species:")

for species in sorted(set(labels)):

    count = labels.count(species)

    print(
        f"{species}: {count} fragment(s)"
    )


# ============================================================
# 3. K-MER FREQUENCY
# ============================================================

def kmer_frequency(sequence, k):

    kmers = [
        sequence[i:i + k]
        for i in range(len(sequence) - k + 1)
    ]

    return Counter(kmers)


# ============================================================
# 4. BUILD K-MER MATRIX
# ============================================================

def build_kmer_matrix(sequences, k):

    all_kmers = set()
    kmer_counts = []

    for sequence in sequences:

        counts = kmer_frequency(
            sequence,
            k
        )

        kmer_counts.append(counts)
        all_kmers.update(counts.keys())

    all_kmers = sorted(all_kmers)

    X = np.zeros(
        (len(sequences), len(all_kmers)),
        dtype=float
    )

    for i, counts in enumerate(kmer_counts):

        total = sum(counts.values())

        for j, kmer in enumerate(all_kmers):

            if total > 0:

                X[i, j] = (
                    counts.get(kmer, 0) / total
                )

    return X


# ============================================================
# 5. PCA ANALYSIS
# ============================================================

def run_pca(X, labels, k):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_scaled)

    explained_variance = (
        pca.explained_variance_ratio_
    )

    print()
    print(f"===== PCA for k={k} =====")

    print(
        f"PC1 explained variance: "
        f"{explained_variance[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{explained_variance[1] * 100:.2f}%"
    )

    unique_labels = sorted(set(labels))

    plt.figure(figsize=(8, 6))

    for lab in unique_labels:

        idx = [
            i
            for i, label in enumerate(labels)
            if label == lab
        ]

        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=lab,
            alpha=0.7
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        f"PCA of Frog DNA k-mer Features (k={k})"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"task3_rf_pca_k{k}.png",
        dpi=300
    )

    plt.show()

    return X_pca


# ============================================================
# 6. RANDOM FOREST CLASSIFICATION
# ============================================================

def run_classification(X, labels, k):

    print()
    print(
        f"===== Random Forest Classification for k={k} ====="
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )
    )

    # Random Forest does not require feature scaling,
    # but the original preprocessing structure is kept.

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    species_labels = sorted(set(labels))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=species_labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=species_labels
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        f"Random Forest Confusion Matrix (k={k})"
    )

    plt.tight_layout()

    plt.savefig(
        f"task3_rf_confusion_matrix_k{k}.png",
        dpi=300
    )

    plt.show()

    return accuracy


# ============================================================
# 7. MAIN ANALYSIS
# ============================================================

results = {}

for k in [3, 4, 5, 6]:

    print()
    print("=" * 60)
    print(f"RANDOM FOREST ANALYSIS FOR k = {k}")
    print("=" * 60)

    X = build_kmer_matrix(
        sequences,
        k
    )

    print(
        f"K-mer feature matrix shape: {X.shape}"
    )

    run_pca(
        X,
        labels,
        k
    )

    accuracy = run_classification(
        X,
        labels,
        k
    )

    results[k] = accuracy


# ============================================================
# 8. SUMMARY
# ============================================================

print()
print("=" * 60)
print("TASK 3 SUMMARY - RANDOM FOREST")
print("=" * 60)

for k, accuracy in results.items():

    print(
        f"k={k}: accuracy = {accuracy:.4f}"
    )


best_k = max(
    results,
    key=results.get
)

print()
print(
    f"Best k according to Random Forest classification accuracy: "
    f"k={best_k}"
)