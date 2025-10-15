# 🛡️ Hate Speech Detection using LSTM

This project builds a deep learning model using **Long Short-Term Memory (LSTM)** networks to detect hate speech and offensive language in tweets.  
The model classifies each tweet into one of three categories:

- **0** → Hate Speech  
- **1** → Offensive Language  
- **2** → Neither  

## 📂 Dataset

The dataset used is [`labeled_data.csv`](labeled_data.csv), which contains the following columns:

| Column                | Description                                      |
|-------------------------|---------------------------------------------------|
| `hate_speech`           | Count of hate speech terms                        |
| `offensive_language`    | Count of offensive language terms                 |
| `neither`               | Neither offensive nor hate                        |
| `class`                 | Label (0 = Hate Speech, 1 = Offensive, 2 = Neither) |
| `tweet`                 | The original tweet text                           |

**Dataset size:** 24,783 tweets

---

## 🧰 Technologies Used

- 🐍 Python 3.10+
- 🧠 TensorFlow / Keras
- 📝 Pandas & NumPy
- 💬 spaCy (for text preprocessing & lemmatization)
- 📊 Scikit-learn
- ⚖️ imbalanced-learn (SMOTE)
- 📈 Matplotlib & Seaborn

---

## 🧹 Data Preprocessing

1. **Remove unwanted columns** (`Unnamed`, `count`, `hate_speech`, etc.)  
2. **Clean tweets** — remove numbers, symbols, URLs, mentions, and special characters.  
3. **Remove extra spaces** using regex.  
4. **Lemmatization** with `spaCy`.  
5. **Stopword removal** to keep only meaningful words.  
6. **One Hot Encoding** of words (`vocab_size = 10000`).  
7. **Padding sequences** to a fixed length of 20 tokens.  
8. **Balance classes** using `SMOTE`.

---

## 🧠 Model Architecture

The model is built with stacked LSTM layers to capture contextual dependencies in text.

