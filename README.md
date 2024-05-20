# README

## Project Overview

This project involves two main tasks: language modeling/regression and language modeling/classification. Each task involves preprocessing a dataset, encoding the data, training models, evaluating the models, and interpreting the results.

### Part 1: Language Modeling / Regression

**Dataset:** [Short Answer Grading Dataset](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/answers.csv)

#### Steps:

1. **Preprocessing NLP Pipeline**:
    - **Tokenization**: Split text into tokens (words).
    - **Stemming**: Reduce words to their root form.
    - **Lemmatization**: Reduce words to their base form using vocabulary and morphological analysis.
    - **Stop Words Removal**: Remove common words that do not contribute significant meaning.
    - **Discretization**: Convert continuous data into discrete bins if necessary.

2. **Data Encoding**:
    - **Word2Vec**: Generate word embeddings using Continuous Bag of Words (CBOW) and Skip Gram models.
    - **Bag of Words (BoW)**: Convert text into fixed-length vectors by counting word occurrences.
    - **TF-IDF**: Transform text into vectors based on Term Frequency-Inverse Document Frequency.

3. **Model Training**:
    - **Support Vector Regression (SVR)**
    - **Naive Bayes Regression**
    - **Linear Regression**
    - **Decision Tree Regression**
    
    *Note: Embeddings will be generated using Word2Vec.*

4. **Model Evaluation**:
    - Evaluate models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and other relevant metrics.
    - Choose the best model based on these metrics and provide a justification for the choice.



### Part 2: Language Modeling / Classification

**Dataset:** [Twitter Entity Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

#### Steps:

1. **Preprocessing NLP Pipeline**:
    - **Tokenization**: Split text into tokens (words).
    - **Stemming**: Reduce words to their root form.
    - **Lemmatization**: Reduce words to their base form using vocabulary and morphological analysis.
    - **Stop Words Removal**: Remove common words that do not contribute significant meaning.
    - **Discretization**: Convert continuous data into discrete bins if necessary.

2. **Data Encoding**:
    - **Word2Vec**: Generate word embeddings using Continuous Bag of Words (CBOW) and Skip Gram models.
    - **Bag of Words (BoW)**: Convert text into fixed-length vectors by counting word occurrences.
    - **TF-IDF**: Transform text into vectors based on Term Frequency-Inverse Document Frequency.

3. **Model Training**:
    - **Support Vector Machine (SVM)**
    - **Naive Bayes Classifier**
    - **Logistic Regression**
    - **Ada Boosting Classifier**

    *Note: Embeddings will be generated using Word2Vec.*

4. **Model Evaluation**:
    - Evaluate models using Accuracy, Loss, F1 Score, and other relevant metrics such as BLEU score.
    - Choose the best model based on these metrics and provide a justification for the choice.

5. **Result Interpretation**:
    - Analyze the performance of each model.
    - Discuss the strengths and weaknesses of the chosen model.
### Conclusion
Through the completion of this project, I have gained several key insights.  Here are the major takeaways: language modeling for regression and language modeling for classification.

