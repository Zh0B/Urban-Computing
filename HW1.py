#this code use chatGPT to help me with generate some plots

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

t100 = pd.read_csv("dataset/288798530_T_T100D_MARKET_ALL_CARRIER.csv", low_memory=False)

#a
Gd = nx.from_pandas_edgelist(t100,
    source="ORIGIN_AIRPORT_ID",
    target="DEST_AIRPORT_ID",
    create_using=nx.DiGraph()
)

Gu = Gd.to_undirected()

components = nx.connected_components(Gu)
components_list = list(components)

sizes = []
for comp in components_list:
    sizes.append(len(comp))
sizes.sort(reverse=True)

nc = len(components_list)
giant_size = sizes[0]
giant_ratio = giant_size / Gu.number_of_nodes()

print(f"Undirected components = {nc}")
print(f"Giant component size = {giant_size} ({giant_ratio:.2%} of all airports)")
print("Top-5 component sizes:", sizes[:5])

plt.bar(range(len(sizes)), sizes)
plt.xlabel("Component rank")
plt.ylabel("Size (# airports)")
plt.title("Connected component sizes (undirected)")
plt.show()

print("Weakly connected components (directed):", nx.number_weakly_connected_components(Gd))
print("Strongly connected components (directed):", nx.number_strongly_connected_components(Gd))

#b
clustering_dict = nx.clustering(Gu)
clustering_values = list(clustering_dict.values())

meanC = np.mean(clustering_values)
medianC = np.median(clustering_values)
stdC = np.std(clustering_values)

print("Average clustering coefficient:", meanC)
print("Median clustering coefficient:", medianC)
print("Standard deviation:", stdC)

plt.hist(clustering_values, bins=25, edgecolor='black')
plt.xlabel("Local clustering coefficient")
plt.ylabel("Number of airports")
plt.title("Distribution of clustering coefficients")
plt.show()

#C
deg_c = nx.degree_centrality(Gu)  
deg_vals = list(deg_c.values())
print("\n[Degree centrality]")
print("mean =", np.mean(deg_vals), "median =", np.median(deg_vals), "std =", np.std(deg_vals))

plt.hist(deg_vals, bins=30, edgecolor="black")
plt.xlabel("Degree centrality")
plt.ylabel("# airports")
plt.title("Distribution of degree centrality")
plt.show()

deg_top = sorted(deg_c.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top-10 (degree):", deg_top)