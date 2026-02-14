import os
import tarfile
import requests
from tqdm import tqdm
import pandas as pd
import kagglehub

# ---------------- CONFIG ----------------
URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
FILENAME = "20news-bydate.tar.gz"
BASE_FOLDER = "20news-bydate"

SPORTS_CATEGORIES = ["rec.sport.baseball", "rec.sport.hockey"]
POLITICS_CATEGORIES = [
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
]

SEPARATOR = "\n'''''\n"


# ---------------- DOWNLOAD 20NEWS ----------------
def download_dataset():
    if os.path.exists(FILENAME):
        print("20News dataset already downloaded.")
        return

    response = requests.get(URL, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(FILENAME, "wb") as file, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

    print("20News download complete.")


# ---------------- EXTRACT ----------------
def extract_dataset():
    if os.path.exists(BASE_FOLDER + "-train"):
        print("20News dataset already extracted.")
        return

    with tarfile.open(FILENAME, "r:gz") as tar:
        tar.extractall()

    print("20News extraction complete.")


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    parts = text.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else text.strip()


# ---------------- COLLECT DOCS ----------------
def collect_docs(split):
    sports_docs, politics_docs = [], []
    split_path = f"{BASE_FOLDER}-{split}"

    for category in SPORTS_CATEGORIES + POLITICS_CATEGORIES:
        category_path = os.path.join(split_path, category)

        for fname in os.listdir(category_path):
            file_path = os.path.join(category_path, fname)
            with open(file_path, "r", encoding="latin1") as f:
                text = clean_text(f.read())

            if category in SPORTS_CATEGORIES:
                sports_docs.append(text + SEPARATOR)
            else:
                politics_docs.append(text + SEPARATOR)

    return sports_docs, politics_docs


# ---------------- SAVE FILES ----------------
def save_files():
    sports_train, politics_train = collect_docs("train")
    sports_test, politics_test = collect_docs("test")

    files = {
        "sports_train.txt": sports_train,
        "sports_test.txt": sports_test,
        "politics_train.txt": politics_train,
        "politics_test.txt": politics_test,
    }

    for filename, content in files.items():
        with open(filename, "w", encoding="utf-8") as f:
            f.write("".join(content))

    print("20News files created.")


# ---------------- AG NEWS PROCESSING ----------------
def process_ag_news():
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")

    train_csv = os.path.join(path, "train.csv")
    test_csv = os.path.join(path, "test.csv")

    def process_file(csv_path, sports_file, politics_file):
        df = pd.read_csv(csv_path, header=None, names=["label", "title", "description"])

        sports_docs = []
        politics_docs = []

        for row in df.itertuples(index=False):
            text = f"{row.title} {row.description}".strip()

            if row.label == '2':
                sports_docs.append(text + SEPARATOR)
            elif row.label == '1':
                politics_docs.append(text + SEPARATOR)

        with open(sports_file, "a" if os.path.exists(sports_file) else "w", encoding="utf-8") as f:
            f.write("".join(sports_docs))

        with open(politics_file, "a" if os.path.exists(politics_file) else "w", encoding="utf-8") as f:
            f.write("".join(politics_docs))

    process_file(train_csv, "sports_train.txt", "politics_train.txt")
    process_file(test_csv, "sports_test.txt", "politics_test.txt")

    print("AG News data appended.")


# ---------------- RUN ----------------
download_dataset()
extract_dataset()
save_files()
process_ag_news()

print("All processing completed successfully.")
