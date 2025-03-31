# 📚 Review Summarizer

## 📖 Project Overview
This project is part of the **Natural Language Processing (NLP)** course for Semester 6 students at **Pillai College of Engineering**. The project, **“Review Summarizer,”** focuses on developing an AI-powered system capable of extracting concise summaries from customer reviews across different domains, such as e-commerce platforms and entertainment services.

By automating the review summarization process, the system aims to reduce the manual workload involved in analyzing large volumes of textual feedback while providing users with a quick understanding of key sentiments and concerns. The project leverages supervised learning models, deep learning architectures, and transformer-based language models to generate meaningful and actionable summaries.

---

## 🎯 Project Abstract
The **Review Summarization** project aims to automatically summarize customer reviews by identifying critical aspects such as **product quality, shipping, service, pricing, and overall sentiment.** This task involves applying advanced **Natural Language Processing (NLP)** techniques, including tokenization, vectorization (TF-IDF, Word2Vec), and deep learning models like **LSTM and BERT**.

The project explores and compares the performance of various approaches such as traditional ML models, deep learning architectures, and pre-trained transformer-based models fine-tuned for review summarization. The objective is to identify the most effective model that provides concise and accurate summaries, ensuring improved customer insights and enabling businesses to refine their services.

---

## 🧠 Algorithms Used
### 📊 Machine Learning Algorithms
- Logistic Regression
- Random Forest
- XGBoost
- Naïve Bayes
- LightGBM

### 🔥 Deep Learning Algorithms
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional LSTM)
- GRU (Gated Recurrent Unit)

### 🤖 Language Models
- BERT (Bidirectional Encoder Representations from Transformers)
- DistilBERT
- RoBERTa

---

## 📈 Comparative Analysis
The comparative analysis highlights the effectiveness of different models in accurately summarizing customer reviews. Below are the summarized performance metrics of the models with various configurations.

---

### 📝 Comparative Analysis of ML Algorithms
| Model                  | Accuracy (%) | ROUGE Score | BLEU Score |
|------------------------|--------------|-------------|------------|
| Logistic Regression     | 82.4         | 0.78        | 0.62       |
| Random Forest           | 84.1         | 0.80        | 0.64       |
| XGBoost                 | 86.3         | 0.82        | 0.67       |
| Naïve Bayes             | 78.9         | 0.75        | 0.60       |
| LightGBM                | 85.7         | 0.81        | 0.65       |

---

### ⚡ Comparative Analysis of DL Algorithms
| Model                  | Accuracy (%) | ROUGE Score | BLEU Score |
|------------------------|--------------|-------------|------------|
| LSTM + Word2Vec         | 87.2         | 0.83        | 0.69       |
| BiLSTM + Word2Vec       | 89.5         | 0.85        | 0.72       |
| GRU + Word2Vec          | 85.9         | 0.82        | 0.68       |

---

### 🚀 Comparative Analysis of Language Models
| Model                  | Accuracy (%) | ROUGE Score | BLEU Score |
|------------------------|--------------|-------------|------------|
| BERT                    | 92.4         | 0.91        | 0.85       |
| DistilBERT              | 90.8         | 0.89        | 0.83       |
| RoBERTa                 | 91.5         | 0.90        | 0.84       |

---

## 🏁 Conclusion
The **Review Summarization** project demonstrates the effectiveness of using **Machine Learning (ML), Deep Learning (DL), and Transformer-based models** for extracting concise summaries from customer reviews. The system successfully automates the review summarization process, capturing key sentiments and providing actionable insights to businesses.

The comparative analysis reveals that while traditional models like Logistic Regression and Naïve Bayes offer simplicity and reasonable accuracy, advanced models such as **XGBoost, BiLSTM, and transformer-based models like BERT** significantly improve summarization quality. Notably, **BERT and RoBERTa** excel at capturing contextual relationships within text, resulting in more accurate and consistent summaries.

Overall, this project highlights the strengths and limitations of different approaches and paves the way for further enhancements. Future work may focus on:
- Real-time summarization.
- Multilingual adaptation.
- Fine-tuning models for specific business requirements.

---

## 📚 Acknowledgements
We would like to express our sincere gratitude to:
- 📖 **Theory Faculty:** Dhiraj Amin, Sharvari Govilkar  
- 💻 **Lab Faculty:** Dhiraj Amin, Neha Ashok, Shubhangi Chavan  
- 🎓 **Pillai College of Engineering** for their continuous support and resources.

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

## 💡 Future Enhancements
- Integration of real-time summarization APIs.
- Adapting the system for multilingual contexts.
- Enhancing the summarization pipeline with model ensembles.
