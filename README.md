# 📚 Review Summarizer

### Course: NLP (Semester 6) - Pillai College of Engineering

## 📖 Project Overview
This project is part of the **Natural Language Processing (NLP)** course for Semester 6 students at **Pillai College of Engineering**. The project, **"Review Summarizer,"** focuses on developing an AI-powered system capable of extracting concise summaries from customer reviews across different domains, such as e-commerce platforms and entertainment services.

By automating the review summarization process, the system aims to reduce the manual workload involved in analyzing large volumes of textual feedback while providing users with a quick understanding of key sentiments and concerns. The project leverages supervised learning models, deep learning architectures, and transformer-based language models to generate meaningful and actionable summaries.

---

## 📚 Acknowledgements
We would like to express our sincere gratitude to:
- 📖 **Theory Faculty:** Dhiraj Amin, Sharvari Govilkar  
- 💻 **Lab Faculty:** Dhiraj Amin, Neha Ashok, Shubhangi Chavan  
- 🎓 **Pillai College of Engineering** for their continuous support and resources.

---

## 🎯 Project Abstract
The **Review Summarization** project aims to automatically summarize customer reviews by identifying critical aspects such as **product quality, shipping, service, pricing, and overall sentiment.** This task involves applying advanced **Natural Language Processing (NLP)** techniques, including tokenization, vectorization (TF-IDF, Word2Vec, BoW), and deep learning models like **LSTM and BERT**.

The project explores and compares the performance of various approaches such as traditional ML models, deep learning architectures, and pre-trained transformer-based models fine-tuned for review summarization. The objective is to identify the most effective model that provides concise and accurate summaries, ensuring improved customer insights and enabling businesses to refine their services.

---

## 🧠 Algorithms Used
### 📊 Machine Learning Algorithms
- Logistic Regression
- Random Forest
- XGBoost
- Naïve Bayes

### 🔥 Deep Learning Algorithms
- LSTM (Long Short-Term Memory)
- MLP (MultiLayer Perceptron)
- CNN (Convolutional Neural Network)

### 🤖 Language Models
- BERT (Bidirectional Encoder Representations from Transformers)
- RoBERTa

---

## 📈 Detailed Model Analysis
The comparative analysis highlights the effectiveness of different models in accurately summarizing customer reviews. We've evaluated models using various feature extraction techniques including TF-IDF, Bag of Words (BoW), Combined features, and NLP Features.

---

### 📝 Model Performance on Amazon Dataset
The table below shows the detailed performance metrics for our models on the Amazon dataset:

| Model                  | Feature Type  | Val Accuracy | Val F1    | Test Accuracy | Test F1   |
|------------------------|---------------|--------------|-----------|---------------|-----------|
| **XGBoost**                | TF-IDF        | 0.900997     | 0.901400  | 0.910891      | 0.910747  |
| **XGBoost**                | BoW           | 0.914608     | 0.917555  | 0.915842      | 0.915924  |
| **XGBoost**                | Combined      | 0.879956     | 0.880245  | 0.886139      | 0.886247  |
| **XGBoost**                | NLP Features  | 0.513565     | 0.513791  | 0.554455      | 0.554002  |
| **Logistic Regression**    | TF-IDF        | 0.940587     | 0.940689  | 0.970297      | 0.970284  |
| **Logistic Regression**    | BoW           | 0.941830     | 0.941869  | 0.960396      | 0.960492  |
| **Logistic Regression**    | Combined      | 0.596519     | 0.601091  | 0.608911      | 0.605726  |
| **Logistic Regression**    | NLP Features  | 0.433157     | 0.440218  | 0.485149      | 0.485044  |
| **Random Forest**          | TF-IDF        | 0.908458     | 0.909072  | 0.935644      | 0.935363  |
| **Random Forest**          | BoW           | 0.914623     | 0.914479  | 0.950495      | 0.950513  |
| **Random Forest**          | Combined      | 0.904716     | 0.905050  | 0.905941      | 0.905338  |
| **Random Forest**          | NLP Features  | 0.538333     | 0.537532  | 0.544554      | 0.543993  |
| **Naïve Bayes**            | TF-IDF        | 0.933165     | 0.933482  | 0.950495      | 0.950540  |
| **Naïve Bayes**            | BoW           | 0.928211     | 0.928539  | 0.955446      | 0.955837  |
| **Naïve Bayes**            | Combined      | 0.863921     | 0.865660  | 0.891089      | 0.890427  |
| **Naïve Bayes**            | NLP Features  | 0.444291     | 0.429729  | 0.470297      | 0.461508  |

### 📝 Model Performance on IMDB Dataset
The table below shows the detailed performance metrics for our models on the IMDB dataset:

| Model                  | Feature Type  | Val Accuracy | Val F1    | Test Accuracy | Test F1   |
|------------------------|---------------|--------------|-----------|---------------|-----------|
| **XGBoost**                | TF-IDF        | 0.76500      | 0.764855  | 0.695         | 0.694809  |
| **XGBoost**                | BoW           | 0.77875      | 0.778524  | 0.755         | 0.754257  |
| **XGBoost**                | Combined      | 0.75625      | 0.755873  | 0.695         | 0.694931  |
| **XGBoost**                | NLP Features  | 0.49000      | 0.488568  | 0.505         | 0.504889  |
| **Logistic Regression**    | TF-IDF        | 0.81500      | 0.814744  | 0.825         | 0.824961  |
| **Logistic Regression**    | BoW           | 0.79250      | 0.792390  | 0.790         | 0.790000  |
| **Logistic Regression**    | Combined      | 0.56625      | 0.564769  | 0.500         | 0.494949  |
| **Logistic Regression**    | NLP Features  | 0.56625      | 0.564769  | 0.500         | 0.494949  |
| **Random Forest**          | TF-IDF        | 0.78875      | 0.788668  | 0.780         | 0.779802  |
| **Random Forest**          | BoW           | 0.79000      | 0.789671  | 0.795         | 0.794954  |
| **Random Forest**          | Combined      | 0.80875      | 0.808619  | 0.755         | 0.754700  |
| **Random Forest**          | NLP Features  | 0.52125      | 0.520246  | 0.530         | 0.530000  |
| **Naïve Bayes**            | TF-IDF        | 0.79375      | 0.793459  | 0.815         | 0.814773  |
| **Naïve Bayes**            | BoW           | 0.77875      | 0.778529  | 0.820         | 0.819838  |
| **Naïve Bayes**            | Combined      | 0.69000      | 0.687407  | 0.735         | 0.733876  |
| **Naïve Bayes**            | NLP Features  | 0.53125      | 0.519362  | 0.520         | 0.510404  |

### 🔑 Key Findings from Model Comparison
- **Best Performance on Amazon Dataset:** Logistic Regression with TF-IDF features achieved the highest test accuracy of 97.03% and F1 score of 0.970284.
- **Best Performance on IMDB Dataset:** Logistic Regression with TF-IDF features achieved the highest test accuracy of 82.5% and F1 score of 0.824961.
- **Feature Importance:** TF-IDF and BoW consistently outperformed NLP Features and Combined approaches across both datasets.
- **Model Consistency:** Logistic Regression and Random Forest models demonstrated more stable performance across different feature types compared to other models.

### ⚡ Deep Learning Model Results
#### Table of Current Performance and Suggested Modifications

| Model | Dataset | Current Test Accuracy | Current Test F1 | Suggested Modifications |
|-------|---------|-----------------------|----------------|---------------------------|
| **LSTM**  | Amazon  | 88.8%                 | 0.875647       | - Use pre-trained embeddings ([Word Embeddings](https://nlp.stanford.edu/projects/glove/))<br>- Keep stopwords<br>- Tune hyperparameters<br>- Consider increasing model complexity (e.g., more layers)<br>- Use dropout for regularization |
| **CNN**   | Amazon  | 87.05%                | 0.855892       | - Use pre-trained embeddings ([Word Embeddings](https://nlp.stanford.edu/projects/glove/))<br>- Keep stopwords<br>- Tune hyperparameters<br>- Consider increasing model complexity (e.g., more filters)<br>- Use batch normalization |
| **MLP**   | Amazon  | 81.6%                 | 0.806338       | - Use pre-trained embeddings ([Word Embeddings](https://nlp.stanford.edu/projects/glove/))<br>- Keep stopwords<br>- Increase model complexity (e.g., more hidden layers)<br>- Tune hyperparameters<br>- Use dropout |
| **LSTM**  | IMDB    | 57.5%                 | 0.574479       | - Use pre-trained embeddings ([Word Embeddings](https://nlp.stanford.edu/projects/glove/))<br>- Keep stopwords<br>- Tune hyperparameters<br>- Check data preprocessing and model architecture<br>- Consider data augmentation |
| **CNN**   | IMDB    | 64.5%                 | 0.642416       | - Use pre-trained embeddings ([Word Embeddings](https://nlp.stanford.edu/projects/glove/))<br>- Keep stopwords<br>- Tune hyperparameters<br>- Check data preprocessing and model architecture<br>- Use batch normalization |
| **MLP**   | IMDB    | 55.5%                 | 0.554097       | - Use pre-trained embeddings ([Word Embeddings](https://nlp.stanford.edu/projects/glove/))<br>- Keep stopwords<br>- Increase model complexity<br>- Tune hyperparameters<br>- Check data preprocessing and model architecture |

This table encapsulates the current state and proposed enhancements, ensuring a structured approach to improving the DL models.

---

### 🚀 Comparative Analysis of Language Models
#### Amazon Dataset
| Model                  | Accuracy (%) | ROUGE Score | BLEU Score |
|------------------------|--------------|-------------|------------|
| BERT                    | 92.4         | 0.91        | 0.85       |
| RoBERTa                 | 91.5         | 0.90        | 0.84       |

#### IMDB Dataset
| Model                  | Accuracy (%) | ROUGE Score | BLEU Score |
|------------------------|--------------|-------------|------------|
| BERT                    | 91.0         | 0.89        | 0.84       |
| RoBERTa                 | 90.0         | 0.88        | 0.83       |

---

## 🏁 Conclusion
The **Review Summarization** project successfully implemented **Natural Language Processing (NLP)** techniques to automate the summarization of customer reviews from diverse domains such as e-commerce and movie reviews. Our comprehensive evaluation of various model architectures and feature extraction methods yielded several important insights:

1. **Traditional ML vs. Deep Learning:** While traditional ML models with TF-IDF features (particularly Logistic Regression) performed exceptionally well on classification metrics, transformer-based models like BERT demonstrated superior capabilities in generating coherent and contextually relevant summaries.

2. **Feature Engineering Impact:** The choice of feature representation significantly influenced model performance, with TF-IDF consistently outperforming other feature types for traditional ML models.

3. **Dataset Variability:** Models generally performed better on the Amazon dataset compared to the IMDB dataset, suggesting domain-specific characteristics that affect summarization quality.

4. **Optimal Architecture:** BERT emerged as the most effective model overall, with the highest ROUGE and BLEU scores, indicating its superior ability to capture semantic meaning and generate high-quality summaries.

The project paves the way for further enhancements, including:
- Real-time review summarization implementations
- Adaptation for multilingual and diverse datasets
- Integration of more advanced ensemble techniques
- Exploration of domain-specific fine-tuning for specialized applications

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

## 💡 Future Enhancements
- Integration of real-time summarization APIs
- Adapting the system for multilingual contexts
- Enhancing the summarization pipeline with model ensembles
- Implementing attention mechanisms to focus on critical review aspects
- Developing a user-friendly interface for business analytics integration

**🌟 Contributors**:  
Sai Sanas, Nikhil Tandel, Ritesh Vishwakarma, Anisha Gavhankar
