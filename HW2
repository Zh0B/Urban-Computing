import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def ccdf(deg):
    d = np.asarray(deg, float)
    d = d[d > 0]
    xs = np.sort(np.unique(d))
    cc = np.array([np.mean(d >= x) for x in xs])
    return xs, cc

def mle_pl_continuous(k, kmin):
    k = np.asarray(k, float)
    k = k[k >= kmin]
    n = len(k)
    if n == 0: return np.nan, 0
    alpha = 1.0 + n / np.sum(np.log(k / (kmin - 0.5)))
    return alpha, n

def ks_ccdf_vs_model(k, kmin, alpha):
    k = np.asarray(k, float)
    k = k[k >= kmin]
    if len(k) == 0: return np.inf
    xs = np.sort(np.unique(k))
    emp = np.array([np.mean(k >= x) for x in xs])
    model = (xs / kmin) ** (1 - alpha)
    return np.max(np.abs(emp - model))

def fit_powerlaw(k, min_tail=30):
    k = np.asarray(k, float)
    k = k[k > 0]
    uniq = np.unique(k)
    if len(uniq) < 5:
        return {'kmin': None, 'alpha': np.nan, 'ks': np.inf, 'n_tail': 0}
    start = int(0.6 * len(uniq))
    best = {'kmin': None, 'alpha': np.nan, 'ks': np.inf, 'n_tail': 0}
    for km in uniq[start:]:
        alpha, n_tail = mle_pl_continuous(k, km)
        if n_tail < min_tail or not np.isfinite(alpha):
            continue
        ks = ks_ccdf_vs_model(k, km, alpha)
        if ks < best['ks']:
            best = {'kmin': float(km), 'alpha': float(alpha), 'ks': float(ks), 'n_tail': int(n_tail)}
    return best

def plot_ccdf_with_fit(deg, title):
    xs, cc = ccdf(deg)
    fit = fit_powerlaw(deg, min_tail=30)
    plt.figure()
    plt.loglog(xs, cc, '.', ms=5)
    if fit['kmin'] is not None and np.isfinite(fit['alpha']):
        line_x = xs[xs >= fit['kmin']]
        model = (line_x / fit['kmin']) ** (1 - fit['alpha'])
        scale = np.mean(np.asarray(deg) >= fit['kmin'])
        plt.loglog(line_x, scale * model, lw=2)
        title += f"  (kmin={fit['kmin']:.0f}, γ≈{fit['alpha']:.2f}, tail n={fit['n_tail']})"
    else:
        title += "  (no stable PL tail)"
    plt.xlabel("degree")
    plt.ylabel("Pr(K ≥ k)")
    plt.title(title)
    plt.tight_layout()

N, m, seed = 6000, 3, 7
G  = nx.barabasi_albert_graph(N, m, seed=seed)
LG = nx.line_graph(G)

plot_ccdf_with_fit([d for _, d in G.degree()],  "BA original")
plot_ccdf_with_fit([d for _, d in LG.degree()], "Line graph of BA")
plt.show()



t100 = pd.read_csv("dataset/288798530_T_T100D_MARKET_ALL_CARRIER.csv", low_memory=False)

df = t100[t100["ORIGIN_AIRPORT_ID"] != t100["DEST_AIRPORT_ID"]]

edges = (df.groupby(["ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID"])
           .size()
           .reset_index(name="WEIGHT")
           .rename(columns={"ORIGIN_AIRPORT_ID":"ORIGIN",
                            "DEST_AIRPORT_ID":"DEST"}))

print(edges.head())


Gd = nx.DiGraph()
for r in edges.itertuples(index=False):
    Gd.add_edge(r.ORIGIN, r.DEST, weight=float(r.WEIGHT))

Gu = Gd.to_undirected()

num_nodes = Gd.number_of_nodes()
num_edges = Gd.number_of_edges()

components = list(nx.connected_components(Gu))
components_sizes = sorted([len(c) for c in components], reverse=True)

giant_size = components_sizes[0]
giant_ratio = giant_size / num_nodes

print(f"Nodes: {num_nodes}, Edges: {num_edges}")
print(f"Undirected components: {len(components)}")
print(f"Giant component size: {giant_size} ({giant_ratio:.2%} of all airports)")
print("Top-5 component sizes:", components_sizes[:5])

print("Weakly connected components:", nx.number_weakly_connected_components(Gd))
print("Strongly connected components:", nx.number_strongly_connected_components(Gd))

G = nx.DiGraph()
for r in edges.itertuples(index=False):
    G.add_edge(r.ORIGIN, r.DEST, weight=float(r.WEIGHT))

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pr = nx.pagerank(G, alpha=0.85, weight="weight")

pr_values = list(pr.values())

print("PageRank summary:")
print("Mean:", np.mean(pr_values))
print("Median:", np.median(pr_values))
print("Std:", np.std(pr_values))
print("Min:", np.min(pr_values), "Max:", np.max(pr_values))

plt.hist(pr_values, bins=50, edgecolor="black")
plt.xlabel("PageRank value")
plt.ylabel("Number of airports")
plt.title("Distribution of PageRank values")
plt.show()

print("(c)")


G_hits = G if isinstance(G, nx.DiGraph) else Gd
assert isinstance(G_hits, nx.DiGraph), "請先建立有向圖 G 或 Gd 後再執行 HITS。"


hubs, auths = nx.hits(G_hits, max_iter=1000, tol=1e-08, normalized=True)


h_vals = np.fromiter(hubs.values(), dtype=float)
a_vals = np.fromiter(auths.values(), dtype=float)

print("\n[HITS summary]")
print("Hub   -> mean:", h_vals.mean(), "median:", np.median(h_vals),
      "std:", h_vals.std(), "min:", h_vals.min(), "max:", h_vals.max())
print("Auth. -> mean:", a_vals.mean(), "median:", np.median(a_vals),
      "std:", a_vals.std(), "min:", a_vals.min(), "max:", a_vals.max())


top10_h = sorted(hubs.items(),  key=lambda x: x[1], reverse=True)[:10]
top10_a = sorted(auths.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop-10 hubs:")
for n, s in top10_h:
    print(n, ":", s)

print("\nTop-10 authorities:")
for n, s in top10_a:
    print(n, ":", s)


plt.figure()
plt.hist(h_vals, bins=50, edgecolor="black")
plt.xlabel("Hub score")
plt.ylabel("Number of nodes")
plt.title("Distribution of HITS Hub Scores")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(a_vals, bins=50, edgecolor="black")
plt.xlabel("Authority score")
plt.ylabel("Number of nodes")
plt.title("Distribution of HITS Authority Scores")
plt.tight_layout()
plt.show()
