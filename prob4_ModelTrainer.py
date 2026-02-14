import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# =========================
# Simple Tokenizer
# =========================
def simple_tokenizer(text):
    # Minimal tokenizer: converts text to lowercase and splits on whitespace
    return text.lower().split()

# =========================
# Data Loading
# =========================
def load_file(path):
    # Reads a text file and separates documents using a custom delimiter
    # Removes empty lines and extra spaces
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    docs = content.split("\n'''''\n")  # Each document is separated by this delimiter
    return [doc.strip().lower() for doc in docs if doc.strip()]

sports_train = load_file("sports_train.txt")
sports_test = load_file("sports_test.txt")
politics_train = load_file("politics_train.txt")
politics_test = load_file("politics_test.txt")

# Combine categories into single training/testing sets
X_train = sports_train + politics_train
y_train = [1]*len(sports_train) + [0]*len(politics_train)  # 1 = Sports, 0 = Politics

X_test = sports_test + politics_test
y_test = [1]*len(sports_test) + [0]*len(politics_test)

# =========================
# Representations
# =========================
# Three vectorization strategies: BoW, TF-IDF, and n-grams (1 to 3)
representations = {
    "1": ("Bag_of_Words", CountVectorizer(tokenizer=simple_tokenizer, lowercase=False)),
    "2": ("TF_IDF", TfidfVectorizer(tokenizer=simple_tokenizer, lowercase=False)),
    "3": ("n_grams_(1,3)", CountVectorizer(ngram_range=(1,3), tokenizer=simple_tokenizer, lowercase=False))
}

# =========================
# Models
# =========================
# Three common classifiers for text data
models = {
    "1": ("Logistic_Regression", LogisticRegression(max_iter=1000)),  # linear classifier
    "2": ("Naive_Bayes", MultinomialNB()),  # probabilistic, works well on word counts
    "3": ("Linear_SVM", LinearSVC())  # strong baseline for high-dimensional sparse vectors
}

# Table to store final test accuracy for each model-representation pair
results_table = pd.DataFrame(index=[v[0] for v in models.values()],
                             columns=[v[0] for v in representations.values()])

# =========================
# Training & Evaluation
# =========================
for rep_key, (rep_name, vectorizer) in representations.items():
    print("\n" + "="*80)
    print(f"Representation Technique: {rep_name}")
    print("="*80)
    
    # Fit vectorizer on training data and transform training/test sets
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    for model_key, (model_name, model) in models.items():
        print("\n" + "-"*60)
        print(f"Model: {model_name}")
        print("-"*60)
        
        # Train model on vectorized training data
        model.fit(X_train_vec, y_train)
        
        # Evaluate performance on both training and test sets
        for dataset, X_vec, y_true in [
            ("Train", X_train_vec, y_train),
            ("Test", X_test_vec, y_test)
        ]:
            
            y_pred = model.predict(X_vec)
            
            # Extract confusion matrix values for detailed error analysis
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Compute standard classification metrics
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            print(f"\n{dataset} Results:")
            print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
            
            # Only test set accuracy is stored for final comparison
            if dataset == "Test":
                results_table.loc[model_name, rep_name] = round(accuracy, 4)

# =========================
# Final Accuracy Table
# =========================
print("\n" + "="*80)
print("Final Test Set Accuracy Comparison Table")
print("="*80)
print(results_table)

# =========================
# User Classification Section
# =========================
print("\n" + "="*80)
print("Document Classification")
print("="*80)

# User selects vectorization and model
print("Choose Representation:")
print("1 - Bag_of_Words")
print("2 - TF_IDF")
print("3 - n_grams_(1,3)")
rep_choice = input("Enter choice (1/2/3): ")

print("\nChoose Model:")
print("1 - Logistic_Regression")
print("2 - Naive_Bayes")
print("3 - Linear_SVM")
model_choice = input("Enter choice (1/2/3): ")

doc_path = input("\nEnter document path: ")

# Load selected vectorizer and model
rep_name, vectorizer = representations[rep_choice]
model_name, model = models[model_choice]

# Retrain the chosen model on full training data
X_train_vec = vectorizer.fit_transform(X_train)
model.fit(X_train_vec, y_train)

with open(doc_path, 'r', encoding='utf-8') as f:
    new_doc = f.read().strip().lower()

# Transform document into the same feature space
new_doc_vec = vectorizer.transform([new_doc])
prediction = model.predict(new_doc_vec)[0]

label = "Sports" if prediction == 1 else "Politics"

print("\n" + "-"*60)
print(f"Representation Used: {rep_name}")
print(f"Model Used: {model_name}")
print(f"Predicted Category: {label}")
print("-"*60)
