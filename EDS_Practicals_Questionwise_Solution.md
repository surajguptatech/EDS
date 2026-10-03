# Elements of Data Science — T.Y.B.Sc. Data Science (Sem 5)
# Question-wise Structured Solutions (Practical 1 to 8)

**Format of every question:**
1. Question (as given in the practical sheet)
2. Code (copy-paste in a Jupyter cell)
3. Answer / Explanation (for the journal)

**Rules**
- Run the cells of one practical **in order, top to bottom** (variables like `df`, `X`, `y` carry forward).
- Change the CSV file name in `pd.read_csv("...")` to match your downloaded file.

## Index
| Practical | Topic | Dataset |
|---|---|---|
| 1 | Data Import and Exploration | Titanic, Lung Cancer |
| 2 | Train–Test Split | Lung Cancer, Insurance |
| 3 | Simple Linear Regression | Insurance, Student Performance |
| 4 | Multiple Linear Regression | Auto MPG |
| 5 | KNN Classification | Iris |
| 6 | Decision Tree | Heart Disease (HeartDisease) |
| 7 | Random Forest | Heart Disease (output) |
| 8 | K-Means Clustering | Mall Customers |

---
---

# PRACTICAL 1 — Data Import and Exploration

## PART A: Titanic Dataset (Q.1)

### Q.1 (a) Import the Titanic dataset
```python
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")      # Kaggle Titanic file
print("Dataset imported successfully")
```
**Answer:** Dataset is loaded in the DataFrame `df`.

### Q.1 (b) Display the first and last five records
```python
print("First 5 records:")
print(df.head())
print("\nLast 5 records:")
print(df.tail())
```
**Answer:** `head()` shows the first 5 rows, `tail()` shows the last 5 rows.

### Q.1 (c) Find the shape of the dataset
```python
print("Shape (rows, columns):", df.shape)
```
**Answer:** Shape gives (number of rows, number of columns). Titanic train data: 891 rows, 12 columns.

### Q.1 (d) Display the column names
```python
print(df.columns.tolist())
```

### Q.1 (e) Display data types of all variables
```python
print(df.dtypes)
```

### Q.1 (f) Generate summary statistics
```python
print(df.describe(include="all"))
```
**Answer:** Shows count, mean, std, min, quartiles, max for numeric columns and count, unique, top, freq for text columns.

### Q.1 (g) Check for missing values
```python
print(df.isnull().sum())
```
**Answer:** `Age`, `Cabin` and `Embarked` contain missing values (Cabin has the most).

### Q.1 (h) Count the number of unique values in each column
```python
print(df.nunique())
```

### Q.1 (i) Number of passengers who survived and did not survive
```python
print(df["Survived"].value_counts())
```
**Answer:** 0 = did not survive, 1 = survived (342 survived, 549 did not in the standard train file).

### Q.1 (j) Average age of passengers
```python
print("Average age:", round(df["Age"].mean(), 2))
```

### Q.1 (k) Average fare paid
```python
print("Average fare:", round(df["Fare"].mean(), 2))
```

### Q.1 (l) Oldest and youngest passenger
```python
oldest = df.loc[df["Age"].idxmax()]
youngest = df.loc[df["Age"].idxmin()]
print("Oldest passenger:\n", oldest[["Name", "Age"]])
print("\nYoungest passenger:\n", youngest[["Name", "Age"]])
```

---

## PART B: Survey Lung Cancer Dataset (Q.2)

### Q.2 (a) Import the dataset
```python
lc = pd.read_csv("survey lung cancer.csv")
```

### Q.2 (b) First and last five records
```python
print(lc.head())
print(lc.tail())
```

### Q.2 (c) Shape of the dataset
```python
print(lc.shape)
```

### Q.2 (d) Column names
```python
print(lc.columns.tolist())
```

### Q.2 (e) Data types
```python
print(lc.dtypes)
```

### Q.2 (f) Summary statistics
```python
print(lc.describe(include="all"))
```

### Q.2 (g) Missing values
```python
print(lc.isnull().sum())
```

### Q.2 (h) Unique values in each column
```python
print(lc.nunique())
```

### Q.2 (i) Frequency of target variable LUNG_CANCER
```python
print(lc["LUNG_CANCER"].value_counts())
```
**Answer:** Shows how many patients have lung cancer (YES) and how many do not (NO). The dataset is imbalanced, with far more YES than NO.

---
---

# PRACTICAL 2 — Train–Test Split

## PART A: Survey Lung Cancer Dataset (Q.1)

### Q.1 (a) Import the dataset
```python
import pandas as pd
from sklearn.model_selection import train_test_split

lc = pd.read_csv("survey lung cancer.csv")
```

### Q.1 (b) Display first five records
```python
print(lc.head())
```

### Q.1 (c) Convert categorical variables into numerical values
```python
lc["GENDER"] = lc["GENDER"].map({"M": 1, "F": 0})
lc["LUNG_CANCER"] = lc["LUNG_CANCER"].map({"YES": 1, "NO": 0})
print(lc.head())
print(lc.dtypes)
```
**Answer:** GENDER (M=1, F=0) and LUNG_CANCER (YES=1, NO=0) are encoded since ML models need numeric data.

### Q.1 (d) Separate independent (X) and dependent (Y) variables
```python
X = lc.drop("LUNG_CANCER", axis=1)
Y = lc["LUNG_CANCER"]
print(X.shape, Y.shape)
```

### Q.1 (e) Perform Train–Test split: 80:20, 70:30, 60:40
```python
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.20, random_state=42)  # 80:20
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.30, random_state=42)  # 70:30
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X, Y, test_size=0.40, random_state=42)  # 60:40
```

### Q.1 (f) Number of observations in training and testing datasets
```python
print("Total:", len(X))
print("80:20 -> Train =", len(X_train1), "Test =", len(X_test1))
print("70:30 -> Train =", len(X_train2), "Test =", len(X_test2))
print("60:40 -> Train =", len(X_train3), "Test =", len(X_test3))
```

### Q.1 (g) Compare the different split ratios
```python
comparison = pd.DataFrame({
    "Split Ratio": ["80:20", "70:30", "60:40"],
    "Training Size": [len(X_train1), len(X_train2), len(X_train3)],
    "Testing Size": [len(X_test1), len(X_test2), len(X_test3)]
})
print(comparison)
```
**Answer:** As the training share goes from 80% to 60%, training data decreases and testing data increases. More training data helps the model learn better; more testing data gives a more reliable evaluation. 80:20 is the most commonly used balance.

---

## PART B: Insurance Dataset (Q.2)

### Q.2 (a) Import the dataset
```python
ins = pd.read_csv("insurance.csv")
```

### Q.2 (b) Display first five records
```python
print(ins.head())
```

### Q.2 (c) Convert categorical variables into numerical values
```python
ins = pd.get_dummies(ins, columns=["sex", "smoker", "region"], drop_first=True)
print(ins.head())
```
**Answer:** `sex`, `smoker`, `region` are categorical and are converted using one-hot encoding.

### Q.2 (d) Separate X and Y
```python
X = ins.drop("charges", axis=1)
Y = ins["charges"]
```

### Q.2 (e) Perform Train–Test split: 80:20, 70:30, 60:40
```python
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.20, random_state=42)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.30, random_state=42)
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X, Y, test_size=0.40, random_state=42)
```

### Q.2 (f) Number of observations in training and testing datasets
```python
print("Total:", len(X))
print("80:20 -> Train =", len(X_train1), "Test =", len(X_test1))
print("70:30 -> Train =", len(X_train2), "Test =", len(X_test2))
print("60:40 -> Train =", len(X_train3), "Test =", len(X_test3))
```

### Q.2 (g) Compare the different split ratios
```python
comparison = pd.DataFrame({
    "Split Ratio": ["80:20", "70:30", "60:40"],
    "Training Size": [len(X_train1), len(X_train2), len(X_train3)],
    "Testing Size": [len(X_test1), len(X_test2), len(X_test3)]
})
print(comparison)
```
**Answer:** Same conclusion as above — larger training share means more data to learn from; larger testing share means more reliable evaluation. 80:20 is the usual choice.

---
---

# PRACTICAL 3 — Simple Linear Regression

## PART A: Insurance Dataset (Q.1)

### Q.1 (a) Import the insurance dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ins = pd.read_csv("insurance.csv")
```

### Q.1 (b) Display first five records
```python
print(ins.head())
```

### Q.1 (c) Find the correlation between (i) bmi & charges (ii) age & charges (iii) children & charges
```python
print("bmi vs charges      :", ins["bmi"].corr(ins["charges"]))
print("age vs charges      :", ins["age"].corr(ins["charges"]))
print("children vs charges :", ins["children"].corr(ins["charges"]))
```
**Answer:** Age has the strongest correlation with charges of the three (roughly 0.30, versus about 0.20 for bmi and 0.07 for children), so it is chosen as the independent variable. Confirm with your own output.

### Q.1 (d) Select independent variable and dependent variable (charges)
```python
X = ins[["age"]]
y = ins["charges"]
```

### Q.1 (e) Split 80:20
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

### Q.1 (f) Train a simple linear regression model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Q.1 (g) Display intercept and coefficient (slope)
```python
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
```
**Answer:** Slope = change in charges for each 1-year increase in age. Intercept = predicted charges at age 0.

### Q.1 (h) Predict charges for the test dataset
```python
y_pred = model.predict(X_test)
```

### Q.1 (i) Prediction error for each observation
```python
error = y_test.values - y_pred
result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred, "Error": error})
print(result.head(10))
```
**Answer:** Error = Actual − Predicted.

### Q.1 (j) Sum of Squared Error (SSE)
```python
SSE = np.sum(error ** 2)
print("SSE:", SSE)
```

### Q.1 (k) Mean Squared Error (MSE)
```python
MSE = SSE / len(y_test)
print("MSE:", MSE)
```

### Q.1 (l) R² score
```python
print("R2 Score:", r2_score(y_test, y_pred))
```
**Answer:** R² shows the proportion of variation in charges explained by age. A low value means age alone is a weak predictor.

### Q.1 (m) Plot the regression line
```python
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Simple Linear Regression: Age vs Charges")
plt.legend()
plt.show()
```

---

## PART B: Student Performance Dataset (Q.2)

### Q.2 (a) Import the dataset
```python
sp = pd.read_csv("Student_Performance.csv")
```

### Q.2 (b) Display first five records
```python
print(sp.head())
```

### Q.2 (c) Correlation with Performance Index
```python
print("Hours Studied          :", sp["Hours Studied"].corr(sp["Performance Index"]))
print("Previous Scores        :", sp["Previous Scores"].corr(sp["Performance Index"]))
print("Sleep Hours            :", sp["Sleep Hours"].corr(sp["Performance Index"]))
print("Sample Question Papers :", sp["Sample Question Papers Practiced"].corr(sp["Performance Index"]))
```
**Answer:** Previous Scores has by far the strongest correlation (about 0.91), so it is chosen.

### Q.2 (d) Select independent variable and dependent variable (Performance Index)
```python
X = sp[["Previous Scores"]]
y = sp["Performance Index"]
```

### Q.2 (e) Split 80:20
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

### Q.2 (f) Train a simple linear regression model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Q.2 (g) Intercept and slope
```python
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
```

### Q.2 (h) Predict Performance Index for the test dataset
```python
y_pred = model.predict(X_test)
```
*(The sheet says "insurance charges" here, but it is a typo — for this dataset we predict Performance Index.)*

### Q.2 (i) Prediction error for each observation
```python
error = y_test.values - y_pred
result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred, "Error": error})
print(result.head(10))
```

### Q.2 (j) SSE
```python
SSE = np.sum(error ** 2)
print("SSE:", SSE)
```

### Q.2 (k) MSE
```python
MSE = SSE / len(y_test)
print("MSE:", MSE)
```

### Q.2 (l) R²
```python
print("R2 Score:", r2_score(y_test, y_pred))
```
**Answer:** A high R² (about 0.84 or more) means Previous Scores explains most of the variation in Performance Index.

### Q.2 (m) Plot the regression line
```python
plt.scatter(X_test, y_test, color="blue", s=10, label="Actual")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Previous Scores")
plt.ylabel("Performance Index")
plt.title("Previous Scores vs Performance Index")
plt.legend()
plt.show()
```

---
---

# PRACTICAL 4 — Multiple Linear Regression (Auto MPG)

### Q.1 (a) Import the dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("auto-mpg.csv")
```

### Q.1 (b) Display first five records
```python
print(df.head())
```

### Q.1 (c) Display shape and data types
```python
print("Shape:", df.shape)
print(df.dtypes)
```

### Q.1 (d) Check for missing values and handle them
```python
# horsepower often has '?' values -> convert to NaN
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
print("Before:\n", df.isnull().sum())

df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
print("After:\n", df.isnull().sum())
```
**Answer:** Missing horsepower values are filled with the median (robust to outliers).

### Q.1 (e) Make the correlation matrix
```python
corr = df.select_dtypes(include=np.number).corr()
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

### Q.1 (f) Select independent variables and dependent variable
```python
X = df[["displacement", "horsepower", "weight", "acceleration"]]
y = df["mpg"]
```

### Q.1 (g) Correlation between independent variables and mpg
```python
print(X.corrwith(y))
```
**Answer:** displacement, horsepower and weight have strong negative correlation with mpg; acceleration has a weak positive correlation.

### Q.1 (h) Split 80:20
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

### Q.1 (i) Build and train Multiple Linear Regression model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Q.1 (j) Display intercept and regression coefficients
```python
print("Intercept:", model.intercept_)
print(pd.DataFrame({"Variable": X.columns, "Coefficient": model.coef_}))
```

### Q.1 (k) Predict mpg for the test dataset
```python
y_pred = model.predict(X_test)
```

### Q.1 (l) DataFrame: Actual MPG, Predicted MPG, Error, Squared Error
```python
result = pd.DataFrame({"Actual MPG": y_test.values, "Predicted MPG": y_pred})
result["Error"] = result["Actual MPG"] - result["Predicted MPG"]
result["Squared Error"] = result["Error"] ** 2
print(result.head(10))
```

### Q.1 (m) Calculate SSE
```python
SSE = result["Squared Error"].sum()
print("SSE:", SSE)
```

### Q.1 (n) Calculate MSE
```python
MSE = SSE / len(result)
print("MSE:", MSE)
```

### Q.1 (o) Calculate R² score
```python
r2 = r2_score(y_test, y_pred)
print("R2:", r2)
```

### Q.1 (p) Calculate Adjusted R²
```python
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R2:", adj_r2)
```
**Formula:** Adjusted R² = 1 − [(1 − R²)(n − 1) / (n − p − 1)]

### Q.1 (q) Interpret the results
**Answer (write in journal, put your own numbers):**
- Intercept is the predicted mpg when all variables are zero.
- Negative coefficients (displacement, horsepower, weight) mean mpg falls as these increase; heavier and more powerful cars give lower mileage.
- SSE and MSE measure the prediction error; smaller is better.
- R² tells what fraction of the variation in mpg the model explains (for example 0.70 = 70%).
- Adjusted R² penalises extra variables; if it is close to R², the variables are useful and the model is not overfitted.

---
---

# PRACTICAL 5 — KNN Classification (Iris)

### Q.1 (a) Import Iris dataset and display first five observations
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

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())
```

### Q.1 (b) Number of observations and variables
```python
print("Observations:", df.shape[0], " Variables:", df.shape[1])
```
**Answer:** 150 observations, 5 variables.

### Q.1 (c) Data types
```python
print(df.dtypes)
```

### Q.1 (d) Missing values
```python
print(df.isnull().sum())
```
**Answer:** No missing values.

### Q.1 (e) Observations per species
```python
print(df["species"].value_counts())
```
**Answer:** 50 each for setosa, versicolor, virginica.

### Q.2 Separate X and y, split 80:20 (random_state = 42)
```python
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape)
```

### Q.3 Standardize the variables and explain why scaling is necessary
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
**Answer:** KNN uses distance (Euclidean) between points. A feature with a larger range dominates the distance and the others are ignored. Standardization (mean 0, std 1) puts all features on the same scale so each contributes equally. Fit the scaler on training data only and then transform the test data, to avoid data leakage.

### Q.4 Build KNN with K = 5, train and predict
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
```

### Q.5 Accuracy, confusion matrix (display + visualize), classification report
```python
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (K = 5)")
plt.show()

print(classification_report(y_test, y_pred))
```
**Answer:** The report shows precision, recall, F1-score for every species plus overall accuracy.

### Q.6 Repeat KNN for K = 1, 3, 5, 7, 9, 11, 13, 15 and store in DataFrame
```python
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train_scaled, y_train)
    accuracies.append(accuracy_score(y_test, m.predict(X_test_scaled)))

results = pd.DataFrame({"K": k_values, "Accuracy": accuracies})
print(results)
```

### Q.7 Plot K vs Accuracy
```python
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K")
plt.ylabel("Testing Accuracy")
plt.title("K vs Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
```
**Observation:** Accuracy changes slightly with K; on Iris it usually stays very high for most values of K.

### Q.8 (a) Accuracy for K = 5
```python
print("Accuracy for K=5:", results.loc[results["K"] == 5, "Accuracy"].values[0])
```
**Answer:** Write the value from your output (commonly 1.00 / 100%).

### Q.8 (b) K with the highest testing accuracy
```python
best = results.loc[results["Accuracy"].idxmax()]
print("Best K:", int(best["K"]), " Accuracy:", best["Accuracy"])
```
**Answer:** The K with the highest accuracy in your table (if tied, choose the smallest K).

### Q.8 (c) Effect of a very small K
**Answer:** With K = 1 the model is very sensitive to noise and outliers and creates a complex, jagged decision boundary. Low bias but high variance, which means **overfitting**.

### Q.8 (d) Effect of a large K
**Answer:** With large K the model averages many neighbours, giving a smooth boundary, but it ignores local patterns and may mix classes. High bias, low variance, which means **underfitting**, and prediction is slower.

### Q.8 (e) Why feature scaling is important in KNN
**Answer:** KNN depends on distance calculation. Without scaling, features with large values dominate the distance and give biased neighbours. Scaling gives each feature equal importance and improves accuracy.

---
---

# PRACTICAL 6 — Decision Tree Classification (Heart Disease)

### Q.1 (a) Import dataset and display first five records
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

df = pd.read_csv("heart.csv")       # file with 'HeartDisease' column
print(df.head())
```

### Q.1 (b) Number of rows and columns
```python
print("Rows:", df.shape[0], " Columns:", df.shape[1])
```

### Q.1 (c) Data types
```python
print(df.dtypes)
```

### Q.1 (d) Missing values
```python
print(df.isnull().sum())
```

### Q.1 (e) Unique values and frequency of HeartDisease
```python
print(df["HeartDisease"].unique())
print(df["HeartDisease"].value_counts())
```
**Answer:** 0 = No heart disease, 1 = Heart disease.

### Q.1 (f) Identify categorical variables
```python
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Categorical variables:", cat_cols)
```
**Answer:** Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope (columns with text values).

### Q.2 (a) Convert categorical variables to numerical form
```python
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(df_encoded.head())
```
**Answer:** One-hot encoding creates a 0/1 column for each category.

### Q.2 (b) Separate X and y (y = HeartDisease)
```python
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]
```

### Q.2 (c) Split 80:20 with random_state = 42
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

### Q.3 Build Decision Tree (Gini), train and predict
```python
dt_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
```

### Q.4 Metrics, classification report and confusion matrix (Gini)
```python
acc_g  = accuracy_score(y_test, y_pred_gini)
prec_g = precision_score(y_test, y_pred_gini)
rec_g  = recall_score(y_test, y_pred_gini)
f1_g   = f1_score(y_test, y_pred_gini)

print("Accuracy :", acc_g)
print("Precision:", prec_g)
print("Recall   :", rec_g)
print("F1-score :", f1_g)
print("\nClassification Report:\n", classification_report(y_test, y_pred_gini))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gini))
```

### Q.5 Build Decision Tree (Entropy), train and predict
```python
dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_ent = dt_entropy.predict(X_test)
```

### Q.6 Metrics, classification report and confusion matrix (Entropy)
```python
acc_e  = accuracy_score(y_test, y_pred_ent)
prec_e = precision_score(y_test, y_pred_ent)
rec_e  = recall_score(y_test, y_pred_ent)
f1_e   = f1_score(y_test, y_pred_ent)

print("Accuracy :", acc_e)
print("Precision:", prec_e)
print("Recall   :", rec_e)
print("F1-score :", f1_e)
print("\nClassification Report:\n", classification_report(y_test, y_pred_ent))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ent))
```

### Q.7 Compare Gini Index and Entropy
```python
comparison = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Accuracy":  [acc_g, acc_e],
    "Precision": [prec_g, prec_e],
    "Recall":    [rec_g, rec_e],
    "F1-Score":  [f1_g, f1_e]
})
print(comparison)
```
**Answer:** The criterion with the higher scores performs better on this test set. Usually both give similar results because both measure node impurity. Gini is slightly faster (no logarithm); Entropy uses information gain. State which one scored higher in your output.

### Q.8 Visualize the trees (feature names, class names, rules, structure)
```python
feature_names = X.columns.tolist()
class_names = ["No Heart Disease", "Heart Disease"]

# Gini tree
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

# Decision rules in text form
print("Rules - Gini:\n", export_text(dt_gini, feature_names=feature_names, max_depth=3))
print("Rules - Entropy:\n", export_text(dt_entropy, feature_names=feature_names, max_depth=3))
```
**Answer:** `max_depth=3` is only for display so the figure stays readable; the trained model is unchanged. Each node shows the splitting condition, impurity, samples, class counts and predicted class.

---
---

# PRACTICAL 7 — Random Forest Classification

> Target column here is **`output`** (Heart Attack dataset). If your file uses another name (e.g. `target`), change it in the code.

### Q.1 (a) Import dataset, separate X and y (y = output)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

df = pd.read_csv("heart.csv")
print(df.head())

X = df.drop("output", axis=1)
y = df["output"]
```

### Q.1 (b) Split 80:20 with random_state = 42
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape)
```

### Q.3 Build Random Forest (n_estimators = 100, random_state = 42), train and predict
*(The sheet has no Q.2, so we continue with Q.3.)*
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

### Q.4 Accuracy, Precision, Recall, F1-score and confusion matrix
```python
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### Q.5 Random Forest with n_estimators = 10, 50, 100, 150, 200
```python
trees = [10, 50, 100, 150, 200]
accuracies = []
for n in trees:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, m.predict(X_test)))

results = pd.DataFrame({"Number of trees": trees, "Accuracy": accuracies})
print(results)
```
**Answer:** Accuracy generally improves from 10 trees and then levels off around 100–200 trees, because averaging many trees reduces variance. Write the number of trees with the highest accuracy from your output.

---
---

# PRACTICAL 8 — K-Means Clustering (Mall Customers)

### Q.1 (a) Import dataset and display first five records
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")
print(df.head())
```

### Q.1 (b) Number of rows and columns
```python
print("Rows:", df.shape[0], " Columns:", df.shape[1])
```

### Q.1 (c) Data types
```python
print(df.dtypes)
```

### Q.1 (d) Missing values
```python
print(df.isnull().sum())
```

### Q.1 (e) Numerical variables suitable for clustering
```python
print(df.select_dtypes(include=np.number).columns.tolist())
```
**Answer:** Age, Annual Income (k$) and Spending Score (1-100) are suitable. CustomerID is just an identifier and is not used.

### Q.2 (a, b) Select Annual Income and Spending Score; create feature matrix X
```python
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
print(X.head())
```

### Q.2 (c) Standardize using StandardScaler
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Q.3 (a, b) Apply K-Means with K = 5 and fit on standardized data
```python
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
kmeans.fit(X_scaled)
```

### Q.3 (c) Assign cluster labels to each customer
```python
df["Cluster"] = kmeans.labels_
```

### Q.3 (d) Display dataset with the assigned cluster
```python
print(df.head(10))
```

### Q.4 (a, b) K-Means for K = 1 to 10 and calculate WCSS
```python
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
```

### Q.4 (c) Table of K and WCSS
```python
elbow_table = pd.DataFrame({"K": range(1, 11), "WCSS": wcss})
print(elbow_table)
```

### Q.4 (d) Plot the Elbow Curve
```python
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
```

### Q.4 (e) Identify appropriate value of K
**Answer:** WCSS falls sharply from K = 1 to K = 5 and then decreases slowly, forming an elbow at **K = 5**. So K = 5 is appropriate.

### Q.5 Plot customer clusters with centroids
```python
centroids = scaler.inverse_transform(kmeans.cluster_centers_)   # back to original scale

plt.figure(figsize=(9, 6))
colors = ["red", "blue", "green", "orange", "purple"]
for i in range(5):
    plt.scatter(df.loc[df["Cluster"] == i, "Annual Income (k$)"],
                df.loc[df["Cluster"] == i, "Spending Score (1-100)"],
                s=50, c=colors[i], label=f"Cluster {i}")
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c="black", marker="X", label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Clusters")
plt.legend()
plt.show()
```

### Q.6 (a) Number of customers in each cluster
```python
print(df["Cluster"].value_counts().sort_index())
```

### Q.6 (b) Average Annual Income and Spending Score for each cluster
```python
summary = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
summary["Customers"] = df["Cluster"].value_counts().sort_index()
print(summary)
```
**Answer (typical interpretation, cluster numbers may differ on your run):**

| Segment | Income | Spending |
|---|---|---|
| Standard customers | Medium | Medium |
| Target customers | High | High |
| Careful customers | High | Low |
| Careless customers | Low | High |
| Sensible customers | Low | Low |

---

## Final checklist before submission
1. Replace every CSV file name with your own file.
2. Practical 6 uses the `HeartDisease` column; Practical 7 uses the `output` column (two different heart datasets).
3. Run each practical top to bottom and take screenshots of the outputs.
4. Write numeric results (accuracy, R², etc.) from your own output.
