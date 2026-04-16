import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os

# --- 3.1. Hiểu và tiền xử lý dữ liệu ---
# Đảm bảo đường dẫn file chính xác
file_path = r'z:\DOCUMENTS\KPDL\DataMining-Homework\Lab6-Phan-cum\shopping_data.csv'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    df = pd.read_csv(file_path)

    print("--- Thông tin bộ dữ liệu ---")
    print(df.info())
    print("\n--- 5 dòng đầu tiên ---")
    print(df.head())

    # --- 3.2. Trực quan hóa phân bố (Annual Income vs Spending Score) ---
    # Cột 3: Annual Income (k$), Cột 4: Spending Score (1-100)
    data = df.iloc[:, [3, 4]].values

    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], color='blue', edgecolors='k', alpha=0.7)
    plt.title("Phân bố dữ liệu: Thu nhập hàng năm vs Điểm chi tiêu")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # --- 3.3. Trực quan hóa quá trình phân cấp bằng Dendrogram ---
    plt.figure(figsize=(12, 7))
    plt.title("Dendrogram - Shopping Data")
    # Sử dụng phương pháp 'ward' để giảm thiểu phương sai trong cụm
    dend = shc.dendrogram(shc.linkage(data, method='ward'))
    plt.axhline(y=150, color='r', linestyle='--')  # Đường cắt minh họa
    plt.xlabel("Số thứ tự điểm dữ liệu")
    plt.ylabel("Khoảng cách Euclidean")
    plt.show()

    # --- 3.4 & 3.5. Lặp k từ 2 đến 6, huấn luyện và trực quan hóa ---
    k_values = range(2, 7)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    print("\n--- Đánh giá mô hình qua các giá trị k ---")
    
    for i, k in enumerate(k_values):
        # Huấn luyện mô hình
        model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = model.fit_predict(data)
        
        # Tính Silhouette Score
        score = silhouette_score(data, labels)
        print(f"Số cụm k = {k}: Silhouette Score = {score:.4f}")

        # Trực quan hóa lên subplot tương ứng
        scatter = axes[i].scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', edgecolors='k', s=50)
        axes[i].set_title(f"k = {k} (Silhouette: {score:.4f})")
        axes[i].set_xlabel("Annual Income (k$)")
        axes[i].set_ylabel("Spending Score (1-100)")
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Ẩn subplot thừa (vị trí cuối cùng nếu k chỉ đến 6 trong lưới 2x3)
    if len(k_values) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()