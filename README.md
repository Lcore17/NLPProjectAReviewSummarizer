# 📚 Review Summarizer - NLP Project  
**Automated Customer Review Classification & Summarization**  

---

## 📖 Project Overview  
This project, part of the **Natural Language Processing (NLP)** course at **Pillai College of Engineering**, focuses on automating the classification and summarization of customer reviews from domains like e-commerce (Amazon) and entertainment (IMDB). The goal is to categorize reviews into themes (e.g., product quality, pricing, sentiment) and generate concise summaries to help businesses extract actionable insights efficiently.  

**Techniques Used**:  
- **NLP Pipelines**: Tokenization, stopword removal, stemming, lemmatization.  
- **Vectorization**: TF-IDF, Word2Vec, BERT embeddings.  
- **Models**: Traditional ML, Deep Learning (LSTM, CNN), and Transformer-based models (BERT, RoBERTa).  

---

## 🎯 Key Objectives  
1. Classify customer reviews into predefined themes (e.g., shipping, pricing, sentiment).  
2. Compare performance of ML, DL, and language models.  
3. Generate concise summaries highlighting key concerns and sentiments.  

---

## 📊 Datasets  
### 1. Amazon Review Dataset  
- **Categories**: Shipping, Pricing, Packaging, Service, Quality.  
- **Example**:  
  | Review | Category |  
  |---|---|  
  | "Paid $199 for climate-controlled shipping..." | Shipping |  

### 2. IMDB Movie Review Dataset  
- **Categories**: Positive, Negative.  
- **Example**:  
  | Review | Sentiment |  
  |---|---|  
  | "I thought this was a wonderful way..." | Positive |  

---

## 🛠️ Text Preprocessing  
- **Tokenization**: Splitting text into words/subwords.  
- **Stopword Removal**: Filtering common words (e.g., "the", "is").  
- **Stemming/Lemmatization**: Reducing words to root forms (e.g., "running" → "run").  
- **Vectorization**:  
  - **TF-IDF**: For traditional ML models.  
  - **Word2Vec/Embeddings**: For DL models.  
  - **BERT Embeddings**: For transformer-based models.  

---

## 🧠 Models Implemented  
### 📈 Machine Learning Models  
- Logistic Regression  
- Random Forest  
- XGBoost  
- Naïve Bayes  

### 🔥 Deep Learning Models  
- LSTM  
- CNN  
- RNN  

### 🤖 Language Models (LM)  
- BERT  
- RoBERTa  
- GPT-2  
- T5  

---

## 📊 Performance Comparison  
### 📝 Machine Learning Models (Amazon Dataset)  
| Model               | Test Accuracy | Test F1-Score |  
|---------------------|---------------|---------------|  
| Logistic Regression | 97.0%         | 97.0%         |  
| Naïve Bayes         | 95.5%         | 95.8%         |  
| Random Forest       | 93.5%         | 93.5%         |  
| XGBoost             | 91.5%         | 91.5%         |  

### ⚡ Deep Learning Models  
| Model    | Test Accuracy | Notes |  
|----------|---------------|-------|  
| LSTM     | 58.9%         | Best for sequential text |  
| CNN      | 64.1%         | Fast but lacks context |  

### 🚀 Language Models  
| Model    | Accuracy | Training Time |  
|----------|----------|---------------|  
| BERT     | 85-95%   | Very High     |  
| RoBERTa  | 88-96%   | Very High     |  

---

## 🏁 Conclusion  
- **Best Performers**:  
  - **ML**: Logistic Regression (97% accuracy).  
  - **LM**: BERT and RoBERTa (95%+ accuracy).  
- **DL Challenges**: Lower accuracy due to dataset size limitations.  
- **Business Impact**: Automated classification enables quick insights into customer feedback.  

---

## 🔮 Future Enhancements  
1. Real-time summarization APIs.  
2. Multilingual support.  
3. Fine-tuning models for domain-specific tasks.  

---

## 📚 Acknowledgements  
- **Faculty**: Dr. Sharvari Govilkar, Prof. Shubhangi Chavan.  
- **Institution**: Pillai College of Engineering, University of Mumbai.  

---

## 📜 License  
MIT License - See [LICENSE.md](LICENSE.md) for details.  

---

**🌟 Contributors**:  
Sai Sanas, Nikhil Tandel, Ritesh Vishwakarma, Anisha Gavhankar  
