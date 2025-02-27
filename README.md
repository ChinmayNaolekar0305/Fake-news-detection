# 📰 Fake News Detection System



## 📌 **Project Overview**
With the rise of misinformation, distinguishing between **real and fake news** has become a crucial challenge. This project builds an **NLP-powered fake news detection system** using:
- **Rule-based approach** (Keyword detection, sentiment analysis)
- **Naïve Bayes + TF-IDF Model** (96% accuracy)
- **BERT Fine-tuned Model** (99.6% accuracy)

## 🛠 **Tech Stack & Tools**
- **Python** (Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn)
- **NLP Libraries** (NLTK, spaCy, TF-IDF)
- **Machine Learning** (Naïve Bayes, Logistic Regression, BERT)
- **Deep Learning** (Hugging Face Transformers, PyTorch/TensorFlow)
- **Data Source**: [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)
- **Jupyter Notebook, Google Colab** for model training

## 📊 **Performance Metrics**
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|------------|---------|------------|
| Rule-Based | 56% | 0.57 | 0.71 | 0.63 |
| Naïve Bayes + TF-IDF | 96% | 0.97 | 0.96 | 0.96 |
| BERT Fine-Tuned | **99.6%** | **0.99** | **0.99** | **0.99** |

## 🔍 **Approach**
### 📌 **1. Rule-Based Approach**
- Uses **100+ predefined keywords** from research papers and real-world fake news samples.
- Implements **VADER Sentiment Analysis** to detect sarcasm and sensationalism.
- Relies on simple **pattern matching**.

### 📌 **2. Machine Learning Approach (Naïve Bayes + TF-IDF)**
- **TF-IDF Vectorization** extracts meaningful word patterns.
- Trained using **Naïve Bayes Classifier**, achieving **96% accuracy**.
- **Lowercasing, stopword removal, tokenization** for text preprocessing.

### 📌 **3. Deep Learning Approach (BERT)**
- Uses **BERT-base-uncased**, fine-tuned on the fake news dataset.
- **Tokenized text with special markers ([CLS], [SEP])**.
- **Trained using Google Colab GPU**, optimized learning rate and batch size.
- Achieved **99.6% accuracy**, outperforming previous models.

## 📈 **Results & Analysis**
### ✅ **Comparison Between Models**
- **Rule-Based:** Transparent but limited, struggles with unseen fake news patterns.
- **Naïve Bayes:** Strong performance but lacks deep contextual understanding.
- **BERT:** Superior accuracy by **learning contextual relationships in text**.

### 🔥 **Heatmaps & Confusion Matrices**
(Include images of confusion matrices and precision-recall curves)
- **Rule Based Approach:**
     - Classification Report: ![Classification Report Rule Based](https://github.com/user-attachments/assets/f197ca20-4291-4d07-bd13-9ca904c1c755)
     - Heatmap: ![Heatmap Rule Based](https://github.com/user-attachments/assets/dad35d59-a51f-4f3d-be09-0b62cac93079)
       
- **Naive Bayes:**
     - Classification Report: ![Classification Report Naive Bayes](https://github.com/user-attachments/assets/30413f17-426d-45fb-acc1-0016bae43344)
     - Heatmap: ![Heatmap NB](https://github.com/user-attachments/assets/2e666bc3-46a0-42c1-a127-1d3ca7f5e8b5)

- **Bert:**
     - Classification Report: ![Classification Report BERT](https://github.com/user-attachments/assets/769a40da-55c4-4ca6-80f9-9912cd6894ec)


## 🔮 **Future Improvements**
- Combine **BERT + Rule-Based Approach** for interpretability.
- Experiment with **Random Forest, Logistic Regression**.
- Specialize in **Medical Fake News Detection**.

## 🚀 **How to Run the Project**
### 🔹 **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model:
   ```bash
   python fake_news_detection.py
   ```
4. To use Jupyter Notebook:
   ```bash
   jupyter notebook fake_news_detection.ipynb
   ```

## 🤝 **Contributors**
- **Chinmay Naolekar**
- **Rudraksh Sharma**
- **Shruti Agrawal**

## 📜 **License**
This project is licensed under the **MIT License**.

## 📩 **Contact**
For any queries, feel free to reach out via LinkedIn or email.

---

### ⭐ **If you find this project useful, don't forget to give it a star on GitHub!** ⭐

