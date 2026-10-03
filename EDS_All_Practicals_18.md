# Elements of Data Science – T.Y.B.Sc. Data Science (Sem 5)
## Complete Solutions: Practical 1 to 8 (Copy-Paste Python Code)

**How to use**
- Run every practical in a separate Jupyter Notebook / Google Colab file.
- Change the CSV file name/path in `pd.read_csv("...")` to match your downloaded dataset.
- Each `# Q.x` comment matches the question number in your practical sheet.

**Common libraries (install once if needed):**
```python
# pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# PRACTICAL 1 – Data Import and Exploration

## Q.1) Titanic Dataset

```python
import pandas as pd
import numpy as np

# a) Import the Titanic dataset
df = pd.read_csv("train.csv")        # Kaggle Titanic file (train.csv / titanic.csv)

# b) First and last five records
print("First 5 records:")
print(df.head())
print("\nLast 5 records:")
print(df.tail())

# c) Shape of the dataset
print("\nShape (rows, columns):", df.shape)

# d) Column names
print("\nColumn names:")
print(df.columns.tolist())

# e) Data types
print("\nData types:")
print(df.dtypes)

# f) Summary statistics
print("\nSummary statistics:")
print(df.describe(include="all"))

# g) Missing values
print("\nMissing values:")
print(df.isnull().sum())

# h) Number of unique values in each column
print("\nUnique values in each column:")
print(df.nunique())

# i) Survived vs Not survived
print("\nSurvival counts (0 = Did not survive, 1 = Survived):")
print(df["Survived"].value_counts())

# j) Average age
print("\nAverage age of passengers:", round(df["Age"].mean(), 2))

# k) Average fare
print("Average fare paid:", round(df["Fare"].mean(), 2))

# l) Oldest and youngest passenger
oldest = df.loc[df["Age"].idxmax()]
youngest = df.loc[df["Age"].idxmin()]
print("\nOldest passenger:")
print(oldest[["Name", "Age"]])
print("\nYoungest passenger:")
print(youngest[["Name", "Age"]])
```

## Q.2) Survey Lung Cancer Dataset

```python
import pandas as pd

# a) Import the dataset
lc = pd.read_csv("survey lung cancer.csv")

# b) First and last five records
print("First 5 records:")
print(lc.head())
print("\nLast 5 records:")
print(lc.tail())

# c) Shape
print("\nShape:", lc.shape)

# d) Column names
print("\nColumn names:")
print(lc.columns.tolist())

# e) Data types
print("\nData types:")
print(lc.dtypes)

# f) Summary statistics
print("\nSummary statistics:")
print(lc.describe(include="all"))

# g) Missing values
print("\nMissing values:")
print(lc.isnull().sum())

# h) Unique values in each column
print("\nUnique values in each column:")
print(lc.nunique())

# i) Frequency of target variable LUNG_CANCER
print("\nFrequency of LUNG_CANCER:")
print(lc["LUNG_CANCER"].value_counts())
```

---

# PRACTICAL 2 – Train–Test Split

## Q.1) Survey Lung Cancer Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# a) Import
lc = pd.read_csv("survey lung cancer.csv")

# b) First five records
print(lc.head())

# c) Convert categorical variables into numerical values
lc["GENDER"] = lc["GENDER"].map({"M": 1, "F": 0})
lc["LUNG_CANCER"] = lc["LUNG_CANCER"].map({"YES": 1, "NO": 0})
print(lc.head())
print(lc.dtypes)

# d) Separate independent (X) and dependent (Y) variables
X = lc.drop("LUNG_CANCER", axis=1)
Y = lc["LUNG_CANCER"]

# e) Train-Test splits
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.20, random_state=42)  # 80:20
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.30, random_state=42)  # 70:30
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X, Y, test_size=0.40, random_state=42)  # 60:40

# f) Number of observations in training and testing sets
print("Total observations:", len(X))
print("80:20 ->  Train =", len(X_train1), " Test =", len(X_test1))
print("70:30 ->  Train =", len(X_train2), " Test =", len(X_test2))
print("60:40 ->  Train =", len(X_train3), " Test =", len(X_test3))

# g) Compare split ratios
comparison = pd.DataFrame({
    "Split Ratio": ["80:20", "70:30", "60:40"],
    "Training Size": [len(X_train1), len(X_train2), len(X_train3)],
    "Testing Size": [len(X_test1), len(X_test2), len(X_test3)]
})
print(comparison)
```

**g) Comparison (write in your journal):**
As the training share decreases from 80% to 60%, the training set becomes smaller and the testing set becomes larger. A larger training set lets the model learn more patterns (usually better learning), while a larger test set gives a more reliable evaluation. 80:20 is the most commonly used balance.

## Q.2) Insurance Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# a) Import
ins = pd.read_csv("insurance.csv")

# b) First five records
print(ins.head())

# c) Convert categorical variables (sex, smoker, region) into numeric
ins = pd.get_dummies(ins, columns=["sex", "smoker", "region"], drop_first=True)
print(ins.head())

# d) Separate X and Y
X = ins.drop("charges", axis=1)
Y = ins["charges"]

# e) Train-Test splits
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.20, random_state=42)  # 80:20
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.30, random_state=42)  # 70:30
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X, Y, test_size=0.40, random_state=42)  # 60:40

# f) Number of observations
print("Total observations:", len(X))
print("80:20 ->  Train =", len(X_train1), " Test =", len(X_test1))
print("70:30 ->  Train =", len(X_train2), " Test =", len(X_test2))
print("60:40 ->  Train =", len(X_train3), " Test =", len(X_test3))

# g) Compare
comparison = pd.DataFrame({
    "Split Ratio": ["80:20", "70:30", "60:40"],
    "Training Size": [len(X_train1), len(X_train2), len(X_train3)],
    "Testing Size": [len(X_test1), len(X_test2), len(X_test3)]
})
print(comparison)
```

---

# PRACTICAL 3 – Simple Linear Regression

## Q.1) Insurance Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# a) Import
ins = pd.read_csv("insurance.csv")

# b) First five records
print(ins.head())

# c) Correlations with charges
print("bmi vs charges      :", ins["bmi"].corr(ins["charges"]))
print("age vs charges      :", ins["age"].corr(ins["charges"]))
print("children vs charges :", ins["children"].corr(ins["charges"]))

# d) Select independent variable = the one with strongest correlation (age)
X = ins[["age"]]
y = ins["charges"]

# e) 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# f) Train simple linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# g) Intercept and slope
print("Intercept:", model.intercept_)
print("Coefficient (slope):", model.coef_[0])

# h) Predict
y_pred = model.predict(X_test)

# i) Prediction error for each observation
error = y_test.values - y_pred
result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred, "Error": error})
print(result.head(10))

# j) SSE
SSE = np.sum(error ** 2)
print("SSE:", SSE)

# k) MSE
MSE = SSE / len(y_test)
print("MSE:", MSE)

# l) R-squared
print("R2 Score:", r2_score(y_test, y_pred))

# m) Plot regression line
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Simple Linear Regression: Age vs Charges")
plt.legend()
plt.show()
```

> Note: `age` is chosen because it has a stronger correlation with `charges` than `bmi` or `children`. Check your printed values; choose the variable with the highest correlation.

## Q.2) Student Performance Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# a) Import
sp = pd.read_csv("Student_Performance.csv")

# b) First five records
print(sp.head())

# c) Correlations with Performance Index
print("Hours Studied              :", sp["Hours Studied"].corr(sp["Performance Index"]))
print("Previous Scores            :", sp["Previous Scores"].corr(sp["Performance Index"]))
print("Sleep Hours                :", sp["Sleep Hours"].corr(sp["Performance Index"]))
print("Sample Question Papers Prac:", sp["Sample Question Papers Practiced"].corr(sp["Performance Index"]))

# d) Select independent variable (strongest correlation = Previous Scores)
X = sp[["Previous Scores"]]
y = sp["Performance Index"]

# e) 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# f) Train model
model = LinearRegression()
model.fit(X_train, y_train)

# g) Intercept and slope
print("Intercept:", model.intercept_)
print("Coefficient (slope):", model.coef_[0])

# h) Predict Performance Index for test data
y_pred = model.predict(X_test)

# i) Prediction error
error = y_test.values - y_pred
result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred, "Error": error})
print(result.head(10))

# j) SSE
SSE = np.sum(error ** 2)
print("SSE:", SSE)

# k) MSE
MSE = SSE / len(y_test)
print("MSE:", MSE)

# l) R-squared
print("R2 Score:", r2_score(y_test, y_pred))

# m) Plot regression line
plt.scatter(X_test, y_test, color="blue", s=10, label="Actual")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Previous Scores")
plt.ylabel("Performance Index")
plt.title("Simple Linear Regression: Previous Scores vs Performance Index")
plt.legend()
plt.show()
```

---

# PRACTICAL 4 – Multiple Linear Regression

## Q.1) Auto MPG Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# a) Import the dataset
df = pd.read_csv("auto-mpg.csv")

# b) First five records
print(df.head())

# c) Shape and data types
print("Shape:", df.shape)
print(df.dtypes)

# d) Missing values and handling them
# (horsepower often contains '?' -> convert to NaN)
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
print(df.isnull().sum())
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())   # fill with median
print("After handling:\n", df.isnull().sum())

# e) Correlation matrix
corr = df.select_dtypes(include=np.number).corr()
print(corr)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# f) Independent and dependent variables
X = df[["displacement", "horsepower", "weight", "acceleration"]]
y = df["mpg"]

# g) Correlation of independent variables with mpg
print(X.corrwith(y))

# h) 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# i) Build and train model
model = LinearRegression()
model.fit(X_train, y_train)

# j) Intercept and coefficients
print("Intercept:", model.intercept_)
coef_df = pd.DataFrame({"Variable": X.columns, "Coefficient": model.coef_})
print(coef_df)

# k) Predict mpg for test data
y_pred = model.predict(X_test)

# l) DataFrame: Actual, Predicted, Error, Squared Error
result = pd.DataFrame({
    "Actual MPG": y_test.values,
    "Predicted MPG": y_pred
})
result["Error"] = result["Actual MPG"] - result["Predicted MPG"]
result["Squared Error"] = result["Error"] ** 2
print(result.head(10))

# m) SSE
SSE = result["Squared Error"].sum()
print("SSE:", SSE)

# n) MSE
MSE = SSE / len(result)
print("MSE:", MSE)

# o) R2 score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# p) Adjusted R2
n = len(y_test)              # number of observations
p = X_test.shape[1]          # number of predictors
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R2:", adj_r2)
```

**q) Interpretation (write in your journal – adjust numbers to your output):**
- The intercept is the predicted mpg when all independent variables are zero.
- Negative coefficients (typically for displacement, horsepower and weight) mean mpg decreases as these increase; heavier, more powerful cars give lower mileage. A coefficient shows the change in mpg for a one-unit rise in that variable keeping others constant.
- SSE and MSE measure prediction error; smaller values indicate better fit.
- R² shows the proportion of variation in mpg explained by the model (e.g., 0.70 = 70%). Adjusted R² corrects R² for the number of predictors; being close to R² means the variables are useful and the model is not overfitted.

---

# PRACTICAL 5 – KNN Classification (Iris)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- Q.1) Import Iris dataset ----------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
# (Or from CSV:  df = pd.read_csv("Iris.csv") and use column 'Species')

# a) First five observations
print(df.head())

# b) Number of observations and variables
print("Observations:", df.shape[0], " Variables:", df.shape[1])

# c) Data types
print(df.dtypes)

# d) Missing values
print(df.isnull().sum())

# e) Observations per species
print(df["species"].value_counts())

# ---------- Q.2) X, y and 80:20 split ----------
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ---------- Q.3) Standardization ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Q.4) KNN with K = 5 ----------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# ---------- Q.5) Accuracy, confusion matrix, report ----------
print("Accuracy (K=5):", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (K = 5)")
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------- Q.6) Different values of K ----------
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, pred))

results = pd.DataFrame({"K": k_values, "Accuracy": accuracies})
print(results)

# ---------- Q.7) K vs Accuracy plot ----------
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K")
plt.ylabel("Testing Accuracy")
plt.title("K vs Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# ---------- Q.8) Based on the results ----------
print("a) Accuracy for K = 5:", results.loc[results["K"] == 5, "Accuracy"].values[0])
best = results.loc[results["Accuracy"].idxmax()]
print("b) Best K:", int(best["K"]), "with accuracy", best["Accuracy"])
```

### Q.3 – Why feature scaling is necessary before KNN
KNN classifies by calculating distances (usually Euclidean) between data points. If features have different ranges/units, the feature with the larger scale dominates the distance and the others are effectively ignored. Standardization (mean = 0, standard deviation = 1) puts all features on the same scale so each contributes equally.

### Q.8 – Written answers
- **a)** Accuracy for K = 5: use your output (commonly ≈ 1.00 / 100% for Iris with random_state = 42).
- **b)** Best K: the K with the highest value in your results table (if several tie, choose the smallest K).
- **c) Very small K (e.g., K = 1):** the model is very sensitive to noise and outliers, creating a complex decision boundary → low bias, high variance → **overfitting**.
- **d) Large K:** the model averages over many neighbours, creating a smooth boundary; it may ignore local patterns and mix classes → high bias, low variance → **underfitting**, and it is slower.
- **e) Feature scaling:** KNN is distance-based, so unscaled features with large ranges dominate the distance and give biased/incorrect neighbours; scaling gives each feature equal importance.

---

# PRACTICAL 6 – Decision Tree Classification

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# ---------- Q.1) Import Heart Disease dataset ----------
df = pd.read_csv("heart.csv")       # Kaggle heart failure dataset with 'HeartDisease' column

# a) First five records
print(df.head())

# b) Rows and columns
print("Rows:", df.shape[0], " Columns:", df.shape[1])

# c) Data types
print(df.dtypes)

# d) Missing values
print(df.isnull().sum())

# e) Unique values and frequency of target
print(df["HeartDisease"].unique())
print(df["HeartDisease"].value_counts())

# f) Categorical variables
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical variables:", cat_cols)

# ---------- Q.2) Preprocessing ----------
# a) Encode categorical variables (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(df_encoded.head())

# b) Separate X and y
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# c) 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ---------- Q.3) Decision Tree – Gini ----------
dt_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

# ---------- Q.4) Metrics for Gini ----------
acc_g  = accuracy_score(y_test, y_pred_gini)
prec_g = precision_score(y_test, y_pred_gini)
rec_g  = recall_score(y_test, y_pred_gini)
f1_g   = f1_score(y_test, y_pred_gini)
print("GINI -> Accuracy:", acc_g, "Precision:", prec_g, "Recall:", rec_g, "F1:", f1_g)
print("\nClassification Report (Gini):\n", classification_report(y_test, y_pred_gini))
print("Confusion Matrix (Gini):\n", confusion_matrix(y_test, y_pred_gini))

# ---------- Q.5) Decision Tree – Entropy ----------
dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_ent = dt_entropy.predict(X_test)

# ---------- Q.6) Metrics for Entropy ----------
acc_e  = accuracy_score(y_test, y_pred_ent)
prec_e = precision_score(y_test, y_pred_ent)
rec_e  = recall_score(y_test, y_pred_ent)
f1_e   = f1_score(y_test, y_pred_ent)
print("ENTROPY -> Accuracy:", acc_e, "Precision:", prec_e, "Recall:", rec_e, "F1:", f1_e)
print("\nClassification Report (Entropy):\n", classification_report(y_test, y_pred_ent))
print("Confusion Matrix (Entropy):\n", confusion_matrix(y_test, y_pred_ent))

# ---------- Q.7) Compare Gini and Entropy ----------
comparison = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Accuracy":  [acc_g, acc_e],
    "Precision": [prec_g, prec_e],
    "Recall":    [rec_g, rec_e],
    "F1-Score":  [f1_g, f1_e]
})
print(comparison)

# ---------- Q.8) Visualize trees ----------
feature_names = X.columns.tolist()
class_names = ["No Heart Disease", "Heart Disease"]

# Gini tree (limited depth for readability in the figure; model itself is unchanged)
plt.figure(figsize=(22, 10))
plot_tree(dt_gini, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, max_depth=3, fontsize=8)
plt.title("Decision Tree - Gini Index")
plt.show()

# Entropy tree
plt.figure(figsize=(22, 10))
plot_tree(dt_entropy, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, max_depth=3, fontsize=8)
plt.title("Decision Tree - Entropy")
plt.show()

# Decision rules (text form)
print("Decision Rules - Gini:\n", export_text(dt_gini, feature_names=feature_names, max_depth=3))
print("Decision Rules - Entropy:\n", export_text(dt_entropy, feature_names=feature_names, max_depth=3))
```

### Q.7 – Comparison conclusion (write in your journal)
Compare the table printed by your code. The criterion with the higher accuracy, precision, recall and F1-score performs better on this test set. Typically both criteria give similar results because they measure impurity in a similar way; Gini is slightly faster to compute (no logarithm), while Entropy (information gain) may sometimes give slightly more balanced trees. State which one scored higher in your output and by how much.

---

# PRACTICAL 7 – Random Forest Classification

> This sheet uses a Heart Disease dataset whose target column is **`output`** (the Kaggle "Heart Attack Analysis & Prediction" dataset, usually `heart.csv` with `output`). If your column name is different (e.g. `target`), change it in the code.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ---------- Q.1) Import dataset ----------
df = pd.read_csv("heart.csv")
print(df.head())

# a) Independent variables (X) and dependent variable (y = Output)
X = df.drop("output", axis=1)
y = df["output"]

# b) 80:20 split with random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ---------- Q.3) Random Forest (n_estimators = 100) ----------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ---------- Q.4) Metrics ----------
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- Q.5) Different numbers of trees ----------
trees = [10, 50, 100, 150, 200]
accuracies = []
for n in trees:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test)))

results = pd.DataFrame({"Number of trees": trees, "Accuracy": accuracies})
print(results)
```

**Observation (write in your journal):** Accuracy usually improves as trees increase from 10 and then stabilises around 100–200 trees, because averaging many trees reduces variance. State the number of trees that gave the highest accuracy in your output.

---

# PRACTICAL 8 – K-Means Clustering (Mall Customers)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------- Q.1) Import and explore ----------
df = pd.read_csv("Mall_Customers.csv")

# a) First five records
print(df.head())

# b) Rows and columns
print("Rows:", df.shape[0], " Columns:", df.shape[1])

# c) Data types
print(df.dtypes)

# d) Missing values
print(df.isnull().sum())

# e) Numerical variables suitable for clustering
print(df.select_dtypes(include=np.number).columns.tolist())
print("Suitable for clustering: Age, Annual Income (k$), Spending Score (1-100)")
print("(CustomerID is only an identifier, so it is NOT used.)")

# ---------- Q.2) Feature selection and scaling ----------
# a, b) Select variables and create feature matrix X
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# c) Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Q.3) K-Means with K = 5 ----------
# a, b) Apply and fit
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
kmeans.fit(X_scaled)

# c) Assign cluster labels
df["Cluster"] = kmeans.labels_

# d) Display dataset with clusters
print(df.head(10))

# ---------- Q.4) Elbow method ----------
# a, b) K = 1 to 10 and WCSS
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# c) Table of K and WCSS
elbow_table = pd.DataFrame({"K": range(1, 11), "WCSS": wcss})
print(elbow_table)

# d) Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# e) Appropriate K
print("From the graph, the elbow (sharp bend) occurs at K = 5, so K = 5 is appropriate.")

# ---------- Q.5) Plot clusters with centroids ----------
# Convert centroids back to original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(9, 6))
colors = ["red", "blue", "green", "orange", "purple"]
for i in range(5):
    plt.scatter(df.loc[df["Cluster"] == i, "Annual Income (k$)"],
                df.loc[df["Cluster"] == i, "Spending Score (1-100)"],
                s=50, c=colors[i], label=f"Cluster {i}")
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c="black",
            marker="X", label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Clusters")
plt.legend()
plt.show()

# ---------- Q.6) Cluster analysis ----------
# a) Number of customers in each cluster
print(df["Cluster"].value_counts().sort_index())

# b) Average income and spending score per cluster
cluster_summary = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
cluster_summary["Customers"] = df["Cluster"].value_counts().sort_index()
print(cluster_summary)
```

**Typical cluster interpretation (labels may be numbered differently on your run):**

| Segment | Income | Spending |
|---|---|---|
| Standard customers | Medium | Medium |
| Target customers | High | High |
| Careful customers | High | Low |
| Careless customers | Low | High |
| Sensible customers | Low | Low |

**Elbow answer (Q.4 e):** The WCSS drops sharply from K = 1 to K = 5 and then decreases slowly, forming an "elbow" at K = 5. Hence K = 5 is the appropriate number of clusters.

---

## Quick notes before submission
1. **Dataset file names:** replace the CSV names with the ones you downloaded.
2. **Practical 6 vs 7:** the two sheets use different heart datasets (`HeartDisease` vs `output` column) – use the matching file for each.
3. Run each practical top-to-bottom and take screenshots of the outputs for your journal.
4. Numeric results (accuracy, R², etc.) depend on your dataset version, so copy them from your own output.
