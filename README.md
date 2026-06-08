<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&pause=1000&color=E74C3C&center=true&vCenter=true&width=700&lines=Quora+Question+Pair+Similarity+%F0%9F%94%8D;Detect+Duplicate+Questions+with+NLP;Binary+Classification+%7C+NLP+%7C+Deep+Learning" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/quora-question-pairs)
[![GitHub stars](https://img.shields.io/github/stars/eddiebrock911/Quora-Question-pair?style=for-the-badge&logo=github&color=E74C3C)](https://github.com/eddiebrock911/Quora-Question-pair/stargazers)
[![License](https://img.shields.io/github/license/eddiebrock911/Quora-Question-pair?style=for-the-badge&color=27AE60)](./LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-🚀%20Try%20Now-FF4B4B?style=for-the-badge)](https://quorakit.onrender.com/)

<br/>

> **Quora Question Pair Similarity** — An NLP & Machine Learning project that predicts whether two Quora questions are semantically duplicate using advanced feature engineering, classical ML models, and deep learning techniques.

</div>

---

## 📌 Table of Contents

- [🌐 Live Demo](#-live-demo)
- [🧠 Problem Statement](#-problem-statement)
- [📊 Dataset Overview](#-dataset-overview)
- [🔬 Approach & Methodology](#-approach--methodology)
- [✨ Features Engineered](#-features-engineered)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [👤 Author](#-author)

---

## 🌐 Live Demo

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-quorakit.onrender.com-FF4B4B?style=for-the-badge)](https://quorakit.onrender.com/)

**👉 [https://quorakit.onrender.com/](https://quorakit.onrender.com/)**

> Enter any two questions and instantly check if they are semantically duplicate — powered by the trained ML model.

</div>

---

## 🧠 Problem Statement

Over **100 million people** visit Quora every month. Many users ask questions with the same intent but worded differently — causing a fragmented experience for both readers and writers.

**Goal:** Build a model that **identifies whether two questions are duplicates** so that:
- ✅ Every unique question exists only once on Quora
- ✅ Readers are directed to the canonical best answer
- ✅ Writers don't have to answer the same question multiple times

> This is a **Binary Classification** problem: given `question1` and `question2`, predict `is_duplicate` → `0` (Not Duplicate) or `1` (Duplicate)

---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| 📦 **Source** | [Kaggle — Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) |
| 📏 **Size** | ~60 MB |
| 🔢 **Rows** | 404,290 question pairs |
| ⚖️ **Class Balance** | 63.08% Not Duplicate · 36.92% Duplicate |
| 🔑 **Total Unique Questions** | 537,933 |

### Schema

```
train.csv
├── id           → Row identifier
├── qid1         → Unique ID of question 1
├── qid2         → Unique ID of question 2
├── question1    → Full text of question 1
├── question2    → Full text of question 2
└── is_duplicate → Target label (0 or 1)
```

---

## 🔬 Approach & Methodology

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │  ├── Class distribution
   │  ├── Duplicate vs unique question analysis
   │  └── Question frequency distribution
   │
   ▼
Text Pre-Processing
   │  ├── Lowercasing & punctuation removal
   │  ├── Stopword removal (NLTK)
   │  └── Tokenization & Lemmatization
   │
   ▼
Feature Engineering
   │  ├── Basic NLP features
   │  ├── Token-based features
   │  └── Fuzzy matching features
   │
   ▼
Vectorization
   │  ├── TF-IDF
   │  ├── TF-IDF + Word2Vec
   │  └── GloVe Embeddings
   │
   ▼
Model Training
   │  ├── Classical ML (Random Forest, XGBoost, etc.)
   │  └── Deep Learning (LSTM / Siamese Networks)
   │
   ▼
Evaluation → Log Loss · Accuracy · F1-Score
```

---

## ✨ Features Engineered

### 🔹 Basic Features
| Feature | Description |
|---|---|
| `q1_len` | Character length of Question 1 |
| `q2_len` | Character length of Question 2 |
| `q1_words` | Word count of Question 1 |
| `q2_words` | Word count of Question 2 |
| `word_common` | Number of common words between Q1 & Q2 |
| `word_total` | Total words in Q1 + Q2 |
| `word_share` | `word_common / word_total` |

### 🔹 Token Features
| Feature | Description |
|---|---|
| `freq_q1` | How frequently Q1 appears in dataset |
| `freq_q2` | How frequently Q2 appears in dataset |
| `q1_q2_intersect` | Token intersection ratio |
| `first_word_eq` | Whether first words match |
| `last_word_eq` | Whether last words match |

### 🔹 Fuzzy Features
| Feature | Description |
|---|---|
| `fuzz_ratio` | Simple fuzzy string match ratio |
| `fuzz_partial_ratio` | Partial fuzzy match |
| `token_sort_ratio` | Sorted token match ratio |
| `token_set_ratio` | Set-based token match ratio |

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-AA4A44?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
pip >= 21.x
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/eddiebrock911/Quora-Question-pair.git

# 2. Navigate into project directory
cd Quora-Question-pair

# 3. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 4. Install all dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Using Kaggle API
kaggle competitions download -c quora-question-pairs
unzip quora-question-pairs.zip -d data/
```

> 📌 You'll need a [Kaggle account](https://www.kaggle.com/) and `kaggle.json` API key configured.

---

## 📁 Project Structure

```
Quora-Question-pair/
├── 📂 data/
│   ├── train.csv               # Training data (from Kaggle)
│   └── test.csv                # Test data
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Feature_Engg.ipynb  # Feature engineering
│   ├── 03_Vectorization.ipynb  # Text vectorization
│   └── 04_Models.ipynb         # Model training & evaluation
│
├── 📂 models/
│   └── best_model.pkl          # Saved trained model
│
├── 📂 utils/
│   ├── preprocess.py           # Text preprocessing utilities
│   └── features.py             # Feature extraction functions
│
├── 📄 requirements.txt         # Python dependencies
├── 📄 .gitignore
└── 📄 README.md                # You are here 📍
```

---

## 📈 Results

| Model | Log Loss | Accuracy | F1 Score |
|---|---|---|---|
| Random Forest | ~0.45 | ~79% | ~0.74 |
| XGBoost | ~0.36 | ~82% | ~0.78 |
| LSTM (Deep Learning) | ~0.29 | ~85% | ~0.82 |
| Siamese LSTM | ~0.25 | ~87% | ~0.85 |

> 📌 Results may vary slightly based on hyperparameter tuning and random seed.

### 📉 Key Insights from EDA

```
✅ Total unique questions   :  537,933
⚠️  Repeated questions (>1x) :  111,780  (20.78%)
🔁 Max question repetitions :  157 times
⚖️  Class imbalance ratio   :  63:37 (Not Dup : Dup)
```

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

```bash
# Step 1: Fork this repository
# Step 2: Create your feature branch
git checkout -b feature/YourFeatureName

# Step 3: Commit your changes (follow Conventional Commits)
git commit -m "feat: add new NLP feature for token overlap"

# Step 4: Push to your branch
git push origin feature/YourFeatureName

# Step 5: Open a Pull Request 🚀
```

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for more information.

---

## 👤 Author

<div align="center">

**eddiebrock911**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/eddiebrock911)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/ankitkumar8252)

*If this project helped you, please ⭐ the repo — it really motivates!*

</div>

---

<div align="center">

### 📚 References & Resources

| Resource | Link |
|---|---|
| 📦 Kaggle Competition | [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) |
| 📝 Quora Engineering Blog | [Semantic Question Matching with Deep Learning](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning) |
| 🏆 Kaggle Winning Solutions | [Top Solutions Dropbox](https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0) |
| 📰 Deep Learning Approach | [Identifying Duplicate Questions — Towards Data Science](https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30) |

<br/>

Made with ❤️ by [eddiebrock911](https://github.com/eddiebrock911)

</div>
