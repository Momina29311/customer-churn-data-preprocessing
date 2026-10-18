# Customer Churn Dataset Preparation for Machine Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Create a suitable raw dataset
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "age": np.random.normal(38, 12, n).round().clip(18, 75),
    "monthly_income": np.random.normal(55000, 15000, n).round(),
    "tenure_years": np.random.normal(5, 3, n).round(1).clip(0, 20),
    "num_products": np.random.randint(1, 5, n),
    "city": np.random.choice(
        ["Lahore", "Karachi", "Islamabad", "Multan"],
        n,
        p=[0.4, 0.3, 0.2, 0.1]
    ),
    "subscription_type": np.random.choice(
        ["Basic", "Standard", "Premium"],
        n,
        p=[0.5, 0.3, 0.2]
    ),
    "churn": np.random.choice([0, 1], n, p=[0.7, 0.3])
})

# Add missing values
df.loc[np.random.choice(n, 25, replace=False), "age"] = np.nan
df.loc[np.random.choice(n, 20, replace=False), "monthly_income"] = np.nan
df.loc[np.random.choice(n, 15, replace=False), "city"] = np.nan

# Add duplicate rows
df = pd.concat([df, df.iloc[:8]], ignore_index=True)

# Add outliers
df.loc[np.random.choice(n, 5, replace=False), "monthly_income"] = (
    df["monthly_income"].max() * 8
)

print("Raw dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 2. Handle duplicates
rows_before = len(df)
df = df.drop_duplicates()
print(f"\nRemoved {rows_before - len(df)} duplicate rows.")

# 3. Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())
df["city"] = df["city"].fillna(df["city"].mode()[0])

print("\nMissing values after handling:\n", df.isnull().sum())

# 4. Handle outliers using IQR capping
numeric_features = ["age", "monthly_income", "tenure_years", "num_products"]

def cap_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)

for col in numeric_features:
    df[col] = cap_outliers(df[col])

# 5. Visualizations
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

df["churn"].value_counts().plot(
    kind="bar", ax=axes[0], color=["steelblue", "orange"]
)
axes[0].set_title("Churn Distribution")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")

df["age"].plot(
    kind="hist", bins=25, ax=axes[1], color="seagreen", edgecolor="black"
)
axes[1].set_title("Age Distribution")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Frequency")

df.boxplot(column="monthly_income", by="subscription_type", ax=axes[2])
axes[2].set_title("Monthly Income by Subscription Type")
axes[2].set_ylabel("Monthly Income")

plt.suptitle("")
plt.tight_layout()
plt.savefig("visualizations.png", dpi=150)
plt.show()

# 6. Encode categorical features and scale numerical features
categorical_features = ["city", "subscription_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features
        )
    ]
)

X = df.drop(columns=["churn"])
y = df["churn"]

X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()

final_df = pd.DataFrame(X_processed, columns=feature_names)
final_df["churn"] = y.values

# 7. Export final ML-ready dataset
final_df.to_csv("ml_ready_customer_churn.csv", index=False)

print("\nFinal ML-ready dataset shape:", final_df.shape)
print("\nFinal columns:\n", final_df.columns.tolist())
print("\nFinal dataset preview:\n", final_df.head())