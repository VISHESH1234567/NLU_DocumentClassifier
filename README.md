# Sports vs Politics Text Classifier

## Setup

1. **Clone the repository**
```bash
git clone <repo-url>
cd <repo-folder>
```

2. **Collect Data**

```bash
python prob4_DataCollector.py
```
This automatically downloads the 20 Newsgroups dataset (20news-bydate.tar.gz).
Then extracts into 20news-bydate-train and 20news-bydate-test.
Then creates sports_train.txt, sports_test.txt, politics_train.txt, politics_test.txt and also appends AG News data to these files.

3. **Train and Evaluate Models**
```bash
python prob4_ModelTrainer.py
```
This trains 9 model-representation combinations (BoW, TF-IDF, n-grams × Logistic Regression, Naive Bayes, Linear SVM) and prints evaluation metrics (precision, recall, accuracy, F1).

4. **Sample document inferencing**
```bash
python prob4_ModelTrainer.py
```
Allows user input to classify a new document into Sports or Politics by taking input of representation technique, ML technique and document path after the 9 models are trained.
