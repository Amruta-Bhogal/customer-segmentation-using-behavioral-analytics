import pandas as pd

# Dataset load karo
df = pd.read_csv("data/Mall_Customers.csv")

# First 5 rows show karo
print(df.head())

import pandas as pd

# Dataset load karo
df = pd.read_csv("data/Mall_Customers.csv")

# First 5 rows
print(df.head())

# Dataset ki information
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

import matplotlib.pyplot as plt

# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Annual Income (k$)"], bins=10)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Spending Score (1-100)"], bins=10)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.show()

# Features select karna
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print(X.head())

from sklearn.cluster import KMeans

# K-Means Model
kmeans = KMeans(n_clusters=5, random_state=42)

# Model train karo
kmeans.fit(X)

print("Model Training Complete")

# Har customer ka cluster number
df["Cluster"] = kmeans.labels_

# Pehle 10 customers dekho
print(df.head(10))

# Customer Segmentation with Cluster Centers

plt.figure(figsize=(10,7))

# Customers
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=80
)

# Cluster Centers
centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    marker="X",
    s=300,
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()

# Cluster Summary
print("\nCluster Summary")
print(df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean())

# Cluster Centers
centers = kmeans.cluster_centers_

print("Cluster Centers:")
print(centers)

print("\nCustomer Personas")

cluster_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "High Spending - Low Income",
    3: "High Income - Low Spending",
    4: "Budget Customers"
}

for cluster, name in cluster_names.items():
    print(f"Cluster {cluster}: {name}")

    print("\nMarketing Recommendations")

recommendations = {
    "Average Customers": "Offer seasonal discounts and regular promotional offers.",
    "Premium Customers": "Provide VIP membership, exclusive offers, and premium products.",
    "High Spending - Low Income": "Give cashback offers, EMI options, and discount coupons.",
    "High Income - Low Spending": "Recommend premium products and personalized marketing campaigns.",
    "Budget Customers": "Offer budget-friendly products and festival discounts."
}

for customer_type, recommendation in recommendations.items():
    print(f"{customer_type}: {recommendation}")