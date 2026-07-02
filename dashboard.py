import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import streamlit as st
import pandas as pd

# Page Title
st.title("Customer Segmentation Dashboard")

# Dataset Load
df = pd.read_csv("data/Mall_Customers.csv")

# Heading
st.header("Dataset Preview")

# First 5 Rows
st.dataframe(df.head())

# Total Customers
st.subheader("Total Customers")
st.write(len(df))
# Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Add Cluster Column
df["Cluster"] = kmeans.labels_

# Create Graph
fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=80
)

# Cluster Centers
centers = kmeans.cluster_centers_

ax.scatter(
    centers[:,0],
    centers[:,1],
    c="red",
    marker="X",
    s=250,
    label="Centroids"
)

ax.set_title("Customer Segmentation")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()

st.header("Customer Segmentation Graph")

st.pyplot(fig, clear_figure=True)
# Cluster Summary
st.header("Cluster Summary")

summary = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()

st.dataframe(summary)
# Customer Personas

st.header("Customer Personas")

st.write("""
**Cluster 0:** Average Customers

**Cluster 1:** Premium Customers

**Cluster 2:** High Spending - Low Income

**Cluster 3:** High Income - Low Spending

**Cluster 4:** Budget Customers
""")
# Marketing Recommendations

st.header("Marketing Recommendations")

st.success("⭐ Premium Customers → Offer VIP membership and exclusive products.")

st.info("💰 High Spending - Low Income → Give cashback offers and discount coupons.")

st.warning("🛍 High Income - Low Spending → Recommend premium products through personalized marketing.")

st.error("💸 Budget Customers → Offer budget-friendly products and festival discounts.")

st.write("📢 Average Customers → Send regular promotional offers.")
st.markdown("---")

st.markdown(
"""
### 🛍 Customer Segmentation using Machine Learning

This dashboard analyzes customer behavior using **K-Means Clustering**.

**Dataset:** Mall Customers

**Algorithm:** K-Means Clustering

**Clusters:** 5
"""
)

st.markdown("---")