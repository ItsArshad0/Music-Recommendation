import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"C:\Users\arsha\OneDrive\Desktop\Projects\Music Recomendation\data.csv")
df = df.sample(n=6000, random_state=42).reset_index(drop=True)
print(df.head())

numerical_feature = [
    'valence', 'danceability', 'energy', 'tempo', 
    'acousticness', 'liveness', 'speechiness', 'instrumentalness'
]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_feature])

train_data,test_data=train_test_split(df_scaled,test_size=0.2,random_state=42)

inertia=[]
k_values=range(1,11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_data)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(10,8))
plt.plot(k_values,inertia, marker='o')
plt.xlabel("Number of Clusters(K)")
plt.ylabel("Interia")
plt.title("Elbow Graph")
plt.show()

optimal_k=5
kmeans=KMeans(n_clusters=optimal_k,random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)
print(df)
print(df['Cluster'].value_counts())
pca=PCA(n_components=2)
pca_results=pca.fit_transform(df_scaled)

plt.figure(figsize=(10,8))
plt.scatter(pca_results[:,0], pca_results[:,1],c=df['Cluster'],cmap='viridis')
plt.title("K-means Clustering")
plt.show()

def recommended_songs(song_name, df, num_recommendations=5):
    if song_name not in df['name'].values:
        return f"'{song_name}' not found in the dataset."

    song_cluster = df[df['name'] == song_name]['Cluster'].values[0]
    same_cluster_songs = df[df['Cluster'] == song_cluster]
    
    song_index = same_cluster_songs[same_cluster_songs['name'] == song_name].index[0] 
    cluster_features = same_cluster_songs[numerical_feature]
    
    simi = cosine_similarity(cluster_features, cluster_features)
    
    similar_songs = np.argsort(simi[song_index])[-(num_recommendations+1):-1][::-1]
    
    recommendations = same_cluster_songs.iloc[similar_songs][['name', 'year', 'artists']]
    
    return recommendations


ip = 'Soul Junction'
recommended_song = recommended_songs(ip, df, num_recommendations=5)

print(f"Songs similar to '{ip}':")
print(recommended_song)

df.to_csv("Clustered_data.csv")