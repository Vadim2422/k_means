import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k, tol=1e-4, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        n_samples, n_features = X.shape

        # 1 Инициализируем центроиды
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False), :]

        for i in range(self.max_iter):
            # 2 Добавляем данные к центроидам
            clusters = [[] for _ in range(self.k)]
            for j, point in enumerate(X):
                idx = np.argmin([np.linalg.norm(point - centroid) for centroid in self.centroids])
                clusters[idx].append(j)

            # 3 Обновление центроидов
            old_centroids = self.centroids.copy()
            for idx, cluster in enumerate(clusters):
                if len(cluster) > 0:
                    self.centroids[idx] = np.mean(X[cluster, :], axis=0)

            # 4 Проверка распределения
            if np.allclose(self.centroids, old_centroids, atol=self.tol):
                self.clusters = clusters
                return

        self.clusters = clusters

    def get_wcss(self, X):
        wcss = sum(
            [np.sum(np.square(X[cluster, :] - self.centroids[idx])) for idx, cluster in enumerate(self.clusters)])
        return wcss

    @staticmethod
    def elbow_method(X, max_k=10):
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(k=k)
            kmeans.fit(X)
            wcss.append(kmeans.get_wcss(X))

        fig, ax = plt.subplots()

        ax.plot(range(1, max_k + 1), wcss)
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
        ax.set_title("Elbow Method")

        plt.show()

        wcss = np.array(wcss)
        k = np.arange(1, max_k + 1)
        wcss_diff = np.diff(wcss) / np.diff(k)
        plt.plot(k[1:], wcss_diff)
        changes = np.diff(np.sign(np.diff(wcss_diff)))
        zero_crossings = np.where(changes == -2)[0] + 2
        return zero_crossings[0] if zero_crossings.size > 0 else max_k

    def plot_clusters(self, X):
        if self.centroids is None or self.clusters is None:
            return

        fig, ax = plt.subplots()

        for idx, cluster in enumerate(self.clusters):
            ax.scatter(X[cluster, 0], X[cluster, 1], label=f"Cluster {idx + 1}")

        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], marker="x", color="black", s=300, linewidth=3)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("KMeans Clustering")

        ax.legend()
        plt.show()


X = np.array(
    [(98, 62), (80, 95), (71, 130), (89, 164), (137, 115), (107, 155), (109, 105), (174, 62), (183, 115), (164, 153),
     (142, 174), (140, 80), (308, 123), (229, 171), (195, 237), (180, 298), (179, 340), (251, 262), (300, 176),
     (346, 178), (311, 237), (291, 283), (254, 340), (215, 308), (239, 223), (281, 207), (283, 156)])
KMeans.elbow_method(X)
kmeans = KMeans(k=int(input("Введите количество кластеров -> ")))
kmeans.fit(X)
kmeans.plot_clusters(X)
