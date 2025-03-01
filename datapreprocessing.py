import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

file_fake = pd.read_csv("fake.csv", encoding = "ISO-8859-1")
file_real = pd.read_csv("true.csv", encoding = "ISO-8859-1")

file_fake = file_fake.drop(["subject", "date"], axis = 1)
file_real = file_real.drop(["subject", "date"], axis = 1)
file_fake["label"] = "fake news"
file_real["label"] = "real news"

final_dataset = pd.concat([file_fake, file_real])

nltk.download('stopwords')
stop = set(stopwords.words('english'))

def preprocess(text):
    clean_text = " "
    text = text.lower()
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    clean_t = clean_text.split()
    filtered_text = []
    for i in clean_t:
        if i not in stop:
            filtered_text.append(i)
    clean_text = " ".join(filtered_text)
    return clean_text

def tokenization(text):
    clean = preprocess(text)
    token = clean.split(" ")
    count = Counter(token)
    return token, count

final_dataset["text_token"], final_dataset["text_token_count"] = zip(*final_dataset["text"].apply(tokenization))