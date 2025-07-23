from sklearn.metrics import confusion_matrix
import numpy as np

#序列恢复率
def sequence_recovery_rate(generated_sequence, true_sequence):
    min_length = min(len(generated_sequence), len(true_sequence))
    generated_sequence = generated_sequence[:min_length]
    true_sequence = true_sequence[:min_length]

    match_count = sum(1 for a, b in zip(generated_sequence, true_sequence) if a == b)
    return match_count / len(true_sequence)


#香农熵
def calculate_shannon_entropy(sequences, MAX_SEQ_LENGTH, PROTEIN_ALPHABET):
    total_entropy = 0
    # 遍历每个位置
    for pos in range(MAX_SEQ_LENGTH):
        amino_acid_count = {aa: 0 for aa in PROTEIN_ALPHABET}
        valid_seqs = 0  # 记录当前位置有氨基酸的序列数量
        # 遍历每个序列
        for seq in sequences:
            if pos < len(seq):
                amino_acid_count[seq[pos]] += 1
                valid_seqs += 1

        if valid_seqs > 0:
            entropy = 0
            for count in amino_acid_count.values():
                if count > 0:
                    p = count / valid_seqs
                    entropy -= p * np.log2(p)
            total_entropy += entropy

    average_entropy = total_entropy / MAX_SEQ_LENGTH
    return average_entropy

#PCA可视化
def combined_clustering_evaluation_and_plot(training_sequences, generated_sequences, n_clusters=5, show_plot=True):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"  # 避免内存泄漏
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    amino_acid_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    num_features = len(amino_acid_dict)

    # 将训练集与生成序列拼接
    all_sequences = training_sequences + generated_sequences
    labels_group = [0]*len(training_sequences) + [1]*len(generated_sequences)  # 训练集=0, 生成序列=1

    # One-Hot + 均值编码
    seq_vectors = []
    for seq in all_sequences:
        one_hot_matrix = np.zeros((len(seq), num_features), dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa in amino_acid_dict:
                one_hot_matrix[i, amino_acid_dict[aa]] = 1.0
        seq_vectors.append(one_hot_matrix.mean(axis=0))

    X = np.array(seq_vectors, dtype=np.float32)

    # 如果去重后不足 n_clusters，则调小 n_clusters
    distinct_points = np.unique(X, axis=0).shape[0]
    if distinct_points < n_clusters:
        print(f"可用聚类数仅 {distinct_points}，自动调整 n_clusters = {distinct_points}")
        n_clusters = distinct_points

    if n_clusters <= 1:
        print("数据中向量重复过多，无法正常聚类或降维")
        return 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # 计算轮廓系数
    score = silhouette_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0

    if show_plot:
        # PCA 可能因数据维度或方差不足而报错，做简单检测
        if X.shape[1] < 2 or np.allclose(X.std(axis=0), 0):
            print("数据维度过低或方差过小，跳过 PCA 可视化")
        else:
            pca = PCA(n_components=2)
            try:
                x_pca = pca.fit_transform(X)
                plt.figure(figsize=(6, 5))

                # 用聚类结果上色，用点形区分训练/生成
                markers = {0: 'o', 1: '^'}
                for g in [0, 1]:
                    idx = [i for i, grp in enumerate(labels_group) if grp == g]
                    plt.scatter(
                        x_pca[idx, 0],
                        x_pca[idx, 1],
                        c=[cluster_labels[i] for i in idx],
                        marker=markers[g],
                        cmap="rainbow",
                        label="train" if g == 0 else "generated",
                    )

                plt.colorbar()
                plt.title(f"Combined Clustering (Silhouette Score = {score:.3f})")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.legend()
                plt.tight_layout()
                plt.show()
            except ValueError as e:
                print("PCA 降维失败:", e)

    return score