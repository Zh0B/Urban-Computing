import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

file1 = "202401-citibike-tripdata_1.csv"
file2 = "202401-citibike-tripdata_2.csv"
df1 = pd.read_csv(file1, low_memory=False)
df2 = pd.read_csv(file2, low_memory=False)
df = pd.concat([df1, df2], ignore_index=True)
'''
coords = df[['start_lat', 'start_lng', 'end_lat', 'end_lng']].dropna()

k = 7
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512)
coords['cluster'] = kmeans.fit_predict(coords)
#print("Inertia (SSE):", kmeans.inertia_)

# === 4. Plot start points by cluster ===
plt.figure(figsize=(8, 6))
plt.scatter(
    coords['start_lng'],
    coords['start_lat'],
    c=coords['cluster'],
    cmap='tab10',
    s=10,
    alpha=0.6
)
plt.title(f"Citibike Trip Clusters (Start Points, k={k})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', alpha=0.3)
#plt.show()

kmeanspp = MiniBatchKMeans(
    n_clusters=7,
    random_state=42,
    batch_size=512,
    init='k-means++'
)
kmeanspp.fit(coords)

#print("SSE (k-means++):", kmeanspp.inertia_)
'''
df['started_at'] = pd.to_datetime(df['started_at'])
df['hour'] = df['started_at'].dt.hour

usage = df.groupby(['start_station_name', 'hour']).size().unstack(fill_value=0)
usage_norm = usage.div(usage.sum(axis=1), axis=0)
usage_norm.columns = usage_norm.columns.astype(str)


#k-means
k = 4
kmeans_std = KMeans(n_clusters=k, init='random', random_state=42, n_init=10)
usage_norm['cluster_std'] = kmeans_std.fit_predict(usage_norm)

#k-means++ 
kmeans_pp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
usage_norm['cluster_pp'] = kmeans_pp.fit_predict(usage_norm)

print(f"SSE (k-means): {kmeans_std.inertia_:.4f}")
print(f"SSE (k-means++): {kmeans_pp.inertia_:.4f}")

plt.figure(figsize=(10, 6))
for c in range(k):
    cluster_mean = usage_norm[usage_norm['cluster_pp'] == c].drop(columns=['cluster_std', 'cluster_pp']).mean()
    plt.plot(cluster_mean.index, cluster_mean.values, label=f'Cluster {c}')

plt.title(f'Average Hourly Usage Pattern per Cluster (k={k}, k-means++)')
plt.xlabel('Hour of Day')
plt.ylabel('Normalized Trip Count')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
