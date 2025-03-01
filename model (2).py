import datapreprocessing as dp
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import nltk
import pandas as pd
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
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Feature Functions
def detect_keywords(text, keywords):
    return sum(1 for k in keywords if k.lower() in text.lower())  # Counts matching keywords

def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

def get_text_length(text):
    return len(text.split())

def excessive_punctuation(text):
    return len(re.findall(r'[!?]{2,}', text))

def count_adjectives(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return len([word for word, tag in pos_tags if tag.startswith('JJ')])

# Keywords List (Manually Expanded)
keywords = [
    "miracle cure", "secret revealed", "shocking truth", "government cover-up", "hidden agenda",
    "banned video", "exposed", "leaked", "fraudulent", "big pharma", "crisis actor", "rigged", 
    "clickbait", "hoax", "conspiracy", "propaganda", "must see", "100% effective", "scandal", 
    "false flag", "anti-vax", "shadow government", "deep state", "secret society", "uprising", 
    "quack remedy", "fake science", "fake news", "hoax alert"
]

# Scoring Function
def rule_based_scoring(text):
    score = 0
    
    # Keywords Matching (Weighted Score)
    keyword_matches = detect_keywords(text, keywords)
    score += keyword_matches * 2  # Keywords have a weight of 2
    
    # Sentiment Analysis
    sentiment = get_sentiment_score(text)
    if sentiment < -0.1:  # Negative sentiment
        score += 1
        
    # Excessive Punctuation
    punct_count = excessive_punctuation(text)
    if punct_count > 1:
        score += 1
        
    # Text Length
    length = get_text_length(text)
    if length < 50:  # Short suspicious text
        score += 1
    
    # Adjective Count
    adj_count = count_adjectives(text)
    if adj_count > 5:  # Too many adjectives (sensationalism)
        score += 1

    # Final Rule: Threshold for "Fake News"
    if score >= 4:  # You can tune this threshold
        return "fake news"
    return "real news"

# Evaluate Dataset
def evaluate_rule_based(df):
    df['is_fake'] = df['text'].apply(rule_based_scoring)
    return df

# Load Data
x = dp.final_dataset["text"]
y = dp.final_dataset["label"]

# Create DataFrame
df = pd.DataFrame({'text': x, 'label': y})

# Apply Rule-Based Scoring
df = evaluate_rule_based(df)

# Evaluate Performance
print("Classification Report:")
print(classification_report(df['label'], df['is_fake']))

accuracy = accuracy_score(df['label'], df['is_fake'])
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cmatrix = confusion_matrix(df['label'], df['is_fake'], labels=["fake news", "real news"])

# Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake News", "Real News"],
            yticklabels=["Fake News", "Real News"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Enhanced Rule-Based Fake News Detection")
plt.show()