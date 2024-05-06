import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class DBSCANAnalyzer:
    def __init__(self, data, labels, eps=0.5, min_samples=5):
        self.data = data
        self.labels = labels
        self.eps = eps
        self.min_samples = min_samples

    def silhouette_score(self):
        score = silhouette_score(self.data, self.labels)
        print("Silhouette Score:", score)