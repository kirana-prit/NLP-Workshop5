import pandas as pd
import numpy as np
import time
import re
import nltk
import string
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering,MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

start_time = time.time()
# Load the dataset
file = 'workshop_km.csv'
rows = []
with open(file, 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split(',', 3)
        while len(parts) < 4:
            parts.append('')
            
        rows.append(parts)

# create data frame
df = pd.DataFrame(rows, columns=[
    "date", "query", "username", "comment"
])
df = df.drop(columns=["query"])
load_time = time.time() - start_time
print(f"Dataset loaded in {load_time:.2f} seconds")
print("Loaded dataset")
print("Shape:", df.shape)

# convert date
df["date"] = pd.to_datetime(df["date"], errors='coerce')
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

#raw features
# sentence count
def count_sentences(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s for s in sentences if s.strip() != ""]
    return len(sentences)

df["sentence_count"] = df["comment"].apply(count_sentences)

# English ratio
def language_ratio(text):
    text = str(text)
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(text)
    return english_chars / total_chars if total_chars > 0 else 0

df["english_ratio"] = df["comment"].apply(language_ratio)

#clean data
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)  # URL
    text = re.sub(r'@\w+', '', text)           # mention
    text = re.sub(r'\d+', '', text)            # numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(tokens)

df["clean_comment"] = df["comment"].apply(clean_text)

#text features
df["word_count"] = df["clean_comment"].apply(lambda x: len(x.split()))
df["avg_words_per_sentence"] = df["word_count"] / df["sentence_count"].replace(0,1)
df["unique_word_count"] = df["clean_comment"].apply(lambda x: len(set(x.split())))
df["lexical_diversity"] = df["unique_word_count"] / df["word_count"].replace(0,1)
df["log_word_count"] = np.log1p(df["word_count"])
df["log_unique_word_count"] = np.log1p(df["unique_word_count"])
#user features
# comments per user
user_comment_count = df["username"].value_counts()
df["comment_count_user"] = df["username"].map(user_comment_count)

#prepare for clustering
df = df.dropna()

features = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "english_ratio",
    "log_word_count",
    "log_unique_word_count",
    "lexical_diversity"
]

df_numeric = df[features]

print("Features used:")
print(df_numeric.columns)

#scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
cluster_time = time.time() - start_time
print(f"Feature engineering completed in {cluster_time:.2f} seconds")

#elbow method
inertia = []
K_range = range(1,8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertia.append(km.inertia_)
    
diff1 = np.diff(inertia)
diff2 = np.diff(diff1)
optimal_k = np.argmax(np.abs(diff2)) + 1

print("Auto-detected k from Elbow:", optimal_k)
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
elbow_time = time.time() - start_time
print(f"Elbow method completed in {elbow_time:.2f} seconds")

#sample
sample_size = min(25000, len(scaled_data))
sample_idx = np.random.choice(len(scaled_data), sample_size, replace=False)

X_sample = scaled_data[sample_idx]
df_sample = df.iloc[sample_idx].copy()

print(f"Sample size used for clustering: {sample_size}")

#K-means clustering
kmeans = MiniBatchKMeans(
    n_clusters=optimal_k,
    batch_size=10000,
    random_state=42
)

df_sample["KMeans_Cluster"] = kmeans.fit_predict(X_sample)

print("Silhouette Score (KMeans - Sample):",
      silhouette_score(X_sample, df_sample["KMeans_Cluster"]))

kmeans_time = time.time() - start_time
print(f"K-means clustering completed in {kmeans_time:.2f} seconds")
#Hierarchical clustering
linked = linkage(X_sample, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Dendrogram (Sample)")
plt.show()

hc = AgglomerativeClustering(n_clusters=optimal_k)
df_sample["HAC_Cluster"] = hc.fit_predict(X_sample)

hac_time = time.time() - start_time
print(f"Hierarchical clustering completed in {hac_time:.2f} seconds")

print("Clustering Complete")
end_time = time.time() - start_time
print(f"Total execution time: {end_time:.2f} seconds")
print(df_sample.head(100))