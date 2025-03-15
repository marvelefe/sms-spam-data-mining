import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter 
from pycaret.classification import *  
from sklearn.feature_extraction.text import TfidfVectorizer
from pycaret.clustering import setup, create_model, plot_model, assign_model

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Drop unnecessary columns (in case there are extra ones)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Checking for missing values
df.isnull().sum()  # Should return 0

# Text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the message column
df['processed_message'] = df['message'].apply(preprocess_text)
print(df.head())


# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=3000)  # You can adjust max_features based on your preference

# Apply TF-IDF transformation
X = tfidf.fit_transform(df['processed_message']).toarray()

y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)  # 1 for spam, 0 for ham

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")




# Count the number of spam and ham messages
label_counts = df['label'].value_counts()

# Plot the distribution of labels
plt.figure(figsize=(6, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='husl')
plt.title('Distribution of Spam vs Ham Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.show()


print(label_counts)


# Calculate message lengths (number of characters)
df['message_length'] = df['message'].apply(len)

# Plot histogram of message lengths
plt.figure(figsize=(8, 5))
sns.histplot(df['message_length'], bins=30, kde=True, color='purple')
plt.title('Distribution of Message Lengths')
plt.xlabel('Message Length (Characters)')
plt.ylabel('Frequency')
plt.show()

# Separate length distributions for spam and ham
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['label'] == 'spam']['message_length'], color='red', label='Spam', kde=True, bins=30)
sns.histplot(data=df[df['label'] == 'ham']['message_length'], color='green', label='Ham', kde=True, bins=30)
plt.title('Message Length Distribution (Spam vs Ham)')
plt.xlabel('Message Length (Characters)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

 

# Generate word cloud for spam messages
spam_words = ' '.join(df[df['label'] == 'spam']['processed_message'])
spam_wordcloud = WordCloud(width=600, height=400, background_color='white').generate(spam_words)

# Generate word cloud for ham messages
ham_words = ' '.join(df[df['label'] == 'ham']['processed_message'])
ham_wordcloud = WordCloud(width=600, height=400, background_color='white').generate(ham_words)

# Plot word clouds
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('Spam Messages Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.title('Ham Messages Word Cloud')
plt.axis('off')

plt.show()


# Extract top 20 words from spam and ham messages
spam_words_list = ' '.join(df[df['label'] == 'spam']['processed_message']).split()
ham_words_list = ' '.join(df[df['label'] == 'ham']['processed_message']).split()

spam_word_freq = Counter(spam_words_list).most_common(20)
ham_word_freq = Counter(ham_words_list).most_common(20)

# Convert to DataFrame for visualization
spam_word_df = pd.DataFrame(spam_word_freq, columns=['Word', 'Frequency'])
ham_word_df = pd.DataFrame(ham_word_freq, columns=['Word', 'Frequency'])

# Plot top 20 words in spam
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Frequency', y='Word', data=spam_word_df, palette='Reds_r')
plt.title('Top 20 Words in Spam Messages')

# Plot top 20 words in ham
plt.subplot(1, 2, 2)
sns.barplot(x='Frequency', y='Word', data=ham_word_df, palette='Greens_r')
plt.title('Top 20 Words in Ham Messages')

plt.show()


# Combine the processed message and target label into a new DataFrame
df_model = pd.DataFrame({'message': df['processed_message'], 'label': df['label']})

# Initialize PyCaret for classification
# Target variable is 'label' (spam or ham)
clf = setup(data=df_model, target='label', 
            preprocess=True,  # PyCaret will handle preprocessing
            train_size=0.8,   # 80% training, 20% test
            session_id=123)   # Set a random seed for reproducibility

# Compare all models to find the best one
best_model = compare_models()

# Print the best model
print(best_model)
# Finalize the best model (train on the entire dataset)
final_model = finalize_model(best_model)


plot_model(final_model, plot='confusion_matrix')  # Confusion matrix
plot_model(final_model, plot='auc')               # ROC AUC curve
plot_model(final_model, plot='feature')           # Feature importance (TF-IDF terms)



# Initialize TfidfVectorizer to convert text into numerical features
tfidf = TfidfVectorizer(max_features=1000)  # Limiting to 1000 features to reduce dimensionality
X_tfidf = tfidf.fit_transform(df['processed_message']).toarray()


df_tfidf = pd.DataFrame(X_tfidf, columns=tfidf.get_feature_names_out())

clustering = setup(data=df_tfidf, preprocess=True, session_id=123)

# Apply K-Means clustering
kmeans_model = create_model('kmeans', num_clusters=2)

# Plot the clusters
plot_model(kmeans_model, plot='cluster')
clustered_data = assign_model(kmeans_model)
df['cluster'] = clustered_data['Cluster']
print(df[['label', 'cluster']].head())