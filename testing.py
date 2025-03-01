import datapreprocessing as dp
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import re
import spacy
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
import nltk
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#This is the TFidf vectorization Approach
temp = TfidfVectorizer(max_features = 10000, ngram_range=(1,2))
x = temp.fit_transform(dp.final_dataset["text"])
y = dp.final_dataset["label"]

x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Naive bayes Approach
classify = MultinomialNB()
classify.fit(x_train_df, y_train_df)

predictions = classify.predict(x_test_df)
acc = metrics.accuracy_score(y_test_df, predictions)
creport = classification_report(y_test_df, predictions)
print("Accuracy:", acc)
print(f"Classification Report:\n{creport}")

matrix = confusion_matrix(y_test_df, predictions)
plt.figure()
sns.heatmap(matrix, annot=True, fmt = "d", xticklabels=['real news', 'fake news'], yticklabels=['real news', 'fake news'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Naive Bayes heatmap')
plt.show()

#Rule-based Approach
# Download necessary NLTK components
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Load spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Rule-Based Function: Keyword Matching
def detect_fake_news(text, keywords):
    detected_keywords = [k for k in keywords if k.lower() in text.lower()]
    return 1 if detected_keywords else 0  # 1 = Fake, 0 = Real

# Sentiment Analysis Feature
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

# Text Length Feature
def get_text_length(text):
    return len(text.split())

# Excessive Punctuation Feature
def excessive_punctuation(text):
    return len(re.findall(r'[!?]{2,}', text))

# Adjective Count (Linguistic Analysis)
def count_adjectives(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return len([word for word, tag in pos_tags if tag.startswith('JJ')])  # Adjectives

# Named Entity Count (NER)
def named_entities_count(text):
    doc = nlp(text)
    return len(doc.ents)

# Combined Rule-Based Detection Function
def classify_news(df, keywords):
    # Apply features
    df['keyword_match'] = df['text'].apply(lambda x: detect_fake_news(x, keywords))
    df['sentiment_score'] = df['text'].apply(get_sentiment_score)
    df['text_length'] = df['text'].apply(get_text_length)
    df['excessive_punct'] = df['text'].apply(excessive_punctuation)
    df['num_adjectives'] = df['text'].apply(count_adjectives)
    df['num_entities'] = df['text'].apply(named_entities_count)
    
    # Final rule: Combine features for classification
    def rule_based_decision(row):
        if row['keyword_match'] == 1 or row['sentiment_score'] < -0.1 or row['excessive_punct'] > 2:
            return "fake news"
        return "real news"
    
    df['is_fake'] = df.apply(rule_based_decision, axis=1)
    return df

# Keywords List
keywords = ["miracle cure", "secret revealed", "shocking truth", "government cover-up", "hidden agenda", 
            "banned video", "exposed", "revealed truth", "censored", "leaked", "proven formula", 
            "breakthrough", "must see", "exclusive report", "conspiracy", "fake news", "hoax", 
            "viral sensation", "scandal", "fraudulent", "insider information", "deep state", "big pharma", 
            "alien cover-up", "100% effective", "secret society", "new world order", "hidden cure", 
            "uprising", "quack remedy", "crisis actor", "rigged", "soros-funded", "biased media", 
            "false flag", "shadow government", "anti-vax", "illuminati", "clickbait", "chemtrails", 
            "propaganda", "mind control", "astroturfing", "sorcery", "fake science", "controlled opposition", 
            "hoax alert", "pay-to-play", "red pill", "rumors", "manipulation", "mainstream media", 
            "fake news stories", "US presidential election"]

# Load Dataset
x = dp.final_dataset["text"]
y = dp.final_dataset["label"]

# Split Data
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x, y, test_size=0.25, random_state=0)

# Create Train and Test DataFrames
train_df = pd.DataFrame({'text': x_train_df, 'label': y_train_df})
test_df = pd.DataFrame({'text': x_test_df, 'label': y_test_df})

# Classify News on Train and Test Sets
train_df = classify_news(train_df, keywords)
test_df = classify_news(test_df, keywords)

# Evaluation Metrics
print("Classification Report for Test Set:")
print(classification_report(test_df['label'], test_df['is_fake']))

# Accuracy
accuracy = accuracy_score(test_df['label'], test_df['is_fake'])
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cmatrix = confusion_matrix(test_df['label'], test_df['is_fake'], labels=["fake news", "real news"])

# Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Fake News", "Real News"], 
            yticklabels=["Fake News", "Real News"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Rule-Based Fake News Detection")
plt.show()
