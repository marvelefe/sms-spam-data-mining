# Analysing Patterns in an SMS Spam Dataset Using Data Mining Techniques

## Project Overview
Course Code: MIT 816

Course Title: Data Mining

This assignment was submitted as part of my coursework requirements for the Master in Information Technology (MIT) program at the University of Lagos, Nigeria. 

---

This project analyzes and predicts patterns in a SMS Spam Collection dataset using two data mining techniques; classification and clustering. The primary goal is to classify SMS messages as spam or ham (legitimate) using text processing, exploratory data analysis (EDA), and machine learning models.

## Dataset
The dataset used is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), which consists of 5,574 SMS messages labeled as either spam or ham.

### Dataset Attributes:
- `v1`: A categorical label indicating whether the message is spam or ham.
- `v2`: The text content of the message.

## Project Structure
```
├── plots/                   # Visualizations in PNG format (word clouds, histograms, Confusion matrix, etc.)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── analysis.py              # Main script 
```

## Installation & Setup

### Prerequisites
- Python 3.x
- Required libraries (install using `pip`):
  ```sh
  pip3 install -r requirements.txt
  ```
- Run the main script:
  ```sh
  python3 analysis.py
  ```

## Data Preprocessing
The dataset is clean and does not contain any missing values in the SMS text or
labels (spam or ham). However, we need to preprocess the text data to handle
inconsistencies:
- Text normalization (converting text to lowercase)
- Remove punctuation and special characters
- Tokenizing the text into words.
- Remove stop words (words that don't provide significant meaning, like "is", "the", etc.).
- Perform stemming using `PorterStemmer`
- Apply TF-IDF (Term Frequency-Inverse Document Frequency) transformation for feature extraction

## Exploratory Data Analysis (EDA)
EDA helps in understanding the distribution of the data, identifying trends, and
visualizing relationships between different features. In our case, we will focus
on the distribution of the target labels (spam vs ham), the characteristics of the
message text (e.g., message length), and the most frequently occurring words.
- Label distribution (spam vs ham). We need to analyze how many messages are
labeled as “spam” and how many as “ham” to check for any class imbalance.

![distibution of spam vs ham messages](/plots/Figure_1816.png)

- Message length analysis. We also explore the length of messages by calculating
the number of characters and words in each message. This is important because
spam messages tend to be either very short (containing promotional phrases) or
very long (to look legitimate).

| Message Length Distribution (hist 1) | Message Length Distribution (hist 2) |
|---------------------------------|---------------------------------|
| ![Message Length Distribution (Spam vs Ham)](/plots/Figure_1-message-length-dist.png) | ![Message Length Distribution (Spam vs Ham)](/plots/Figure_1message-lengths.png) |

- Word cloud visualization for spam and ham messages. We create separate word clouds for “spam” and “ham” messages to
understand the common vocabulary used in each category

![Spam and Ham Word Clouds](/plots/Figure_1wordcloud.png)

- Top words frequency analysis. We extract the most frequent words in both spam and ham messages after
preprocessing and this provides insight into the content of spam messages versus
legitimate ones

![Top Words in Spam vs Ham](/plots/Figure_1freequent-words.png)


## Pattern Discovery
We apply two data mining techniques to the SMS Spam Collection Dataset. Classification to predict whether a message is spam or ham and clustering to explore natural groupings in the dataset, even though we already
know the labels.

### Classification using Logistic Regression: 
Using PyCaret, we set up the dataset, train multiple classification models, compare their performance,
and select the best model for spam detection. Once the models are
compared, PyCaret automatically ranks them based on performance
metrics such as accuracy, precision, recall, and F1-score. 

### Clustering with K-Means: 
We also perform K-Means Clustering to explore the possibility of identifying natural clusters in the data. While we already have labels (spam or ham), clustering allows us to examine if the messages group naturally into distinct categories.



## Results & Insights
- The best-performing model was a Logistic regression model and had an accuracy of **89%**, precision of **90%**, and recall of **89%**.

![The evaluated classification models and their results](/plots/Screenshot.png)

- The confusion matrix shows that the model is very precise (**100%**), meaning when it classifies a message as spam, it is never wrong but the recall is also very low (**26.85%**), meaning the model fails to detect a large portion of actual spam messages (high false negative rate).

![The evaluated classification models and their results](/plots/Logistic-regression-confusion-matrix.png)

- The ROC AUC Curve also demonstrates the final model’s ability to distinguish between spam and ham. The AUC-ROC score was **0.91**, indicating strong classification capability.

![The evaluated classification models and their results](/plots/ROC-curves.png)

- Clustering results confirmed that messages naturally group into two distinct categories by the spam/ham characteristics found in the dataset. Cluster **0** represents **Ham** and Cluster **1** represents **spam**.

![The K-means algorithm clusters](/plots/newplot1.png)





## Conclusion
Our results reveal that the model effectively classifies legitimate messages (high
precision) but struggles with identifying spam (low recall). While the ROC
curve shows strong overall performance (AUC of 0.91), the model's recall for
spam detection needs improvement. The results of our K-means clustering also
show that there is a distinction between spam and legitimate messages, and this
is also visible in the word cloud with spam messages often containing
promotional or urgent language like "text", "call" and "free". The logistic
regression model performed quite well but will need to be further trained to be
considered useful in real-world applications such as spam filters, etc.

## Future Direction
- Improve recall by using **ensemble methods** like Random Forest or XGBoost.  
- Use **deep learning** (LSTMs, Transformers) for better text representation.

## Author
[Efe Omoregie](https://github.com/marvelefe)


