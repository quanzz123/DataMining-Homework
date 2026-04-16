import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu
df = pd.read_csv("Mall_Customers.csv")

# 2. Chọn đặc trưng để phân cụm
features = ['Age', 'Annual_Income', 'Spending_Score']
X = df[features]

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Áp dụng K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=42, metric='euclidean')
kmedoids.fit(X_scaled)

# 5. Gán nhãn cụm vào dataframe
df['Cluster'] = kmedoids.labels_

# 6. Hiển thị kết quả 5 dong
print(df[['CustomerID', 'Cluster']].head())

print("Labels:", kmedoids.labels_)
print("Medoids:", kmedoids.cluster_centers_)



# 7. Vẽ biểu đồ phân cụm (Age vs SpendingScore)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df['Age'], y=df['Spending_Score'],
    hue=df['Cluster'], palette='Set2', s=100
)
plt.title('Phân cụm khách hàng bằng K-Medoids')
plt.xlabel('Tuổi')
plt.ylabel('Điểm chi tiêu')
plt.legend(title='Cụm')
plt.show()


# 8. Vẽ biểu đồ phân cụm (Income vs SpendingScore)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df['Annual_Income'], y=df['Spending_Score'],
    hue=df['Cluster'], palette='Set2', s=100
)
plt.title('Phân cụm khách hàng bằng K-Medoids')
plt.xlabel('Thu nhập hàng năm')
plt.ylabel('Điểm chi tiêu')
plt.legend(title='Cụm')
plt.show()
