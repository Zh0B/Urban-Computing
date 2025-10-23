import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
'''
df.info()
df.describe()
print(df.head())
print(df.isnull().sum())
'''
df['holiday'] = df['holiday'].fillna('None')
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['weekday'] = df['date_time'].dt.dayofweek #Mon=0, Sun=6
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['temp_C'] = df['temp'] - 273.15
df = df[(df['temp_C'] > -50) & (df['temp_C'] < 60)]

'''
#time
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='hour', y='traffic_volume', ci=None)
plt.title('Average Traffic Volume by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Traffic Volume')
plt.grid(True)
plt.show()

#weekly
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='weekday', y='traffic_volume')
plt.title('Traffic Volume by Weekday')
plt.xlabel('Weekday (0=Mon, 6=Sun)')
plt.ylabel('Traffic Volume')
plt.show()

#month
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='month', y='traffic_volume')
plt.title('Traffic Volume by Month')
plt.xlabel('Month')
plt.ylabel('Traffic Volume')
plt.show()

#temperature
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='temp_C', y='traffic_volume', alpha=0.3)
sns.lineplot(data=df, x='temp_C', y='traffic_volume', color='red', ci=None)
plt.title('Traffic Volume vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Traffic Volume')
plt.show()

#rain
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='rain_1h', y='traffic_volume', alpha=0.3)
plt.title('Traffic Volume vs Rain (mm)')
plt.xlabel('Rain in Last Hour (mm)')
plt.ylabel('Traffic Volume')
plt.show()

#snow
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='snow_1h', y='traffic_volume', alpha=0.3)
plt.title('Traffic Volume vs Snow (mm)')
plt.xlabel('Snow in Last Hour (mm)')
plt.ylabel('Traffic Volume')
plt.show()

#cloud
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='clouds_all', y='traffic_volume', alpha=0.2)
sns.regplot(data=df, x='clouds_all', y='traffic_volume', scatter=False, color='red', lowess=True)
plt.title('Traffic Volume vs Cloud Coverage (%)')
plt.xlabel('Cloud Coverage (%)')
plt.ylabel('Traffic Volume')
plt.show()

#Weather_main
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='weather_main', y='traffic_volume')
plt.title('Traffic Volume by Main Weather Type')
plt.xlabel('Weather Type')
plt.ylabel('Traffic Volume')
plt.xticks(rotation=45)
plt.show()

#Holiday
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='holiday', y='traffic_volume')
plt.title('Traffic Volume on Holidays vs Normal Days')
plt.xlabel('Holiday')
plt.ylabel('Traffic Volume')
plt.xticks(rotation=45)
plt.show()
'''

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['is_holiday'] = (df['holiday'] != 'None').astype(int)
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

'''
features = ['hour_sin', 'hour_cos', 'weekday', 'is_holiday', 'is_weekend']
target = 'traffic_volume'

train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # manual RMSE calculation
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])

    print(f"{name}:\n  MAE = {mae:.2f}\n  RMSE = {rmse:.2f}\n  R² = {r2:.3f}\n")

results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R2'])
print(results_df)


for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title(f"{name} - Predicted vs Actual")
    plt.show()

from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kfold)
    print(f"{name} 5-fold CV R² mean: {scores.mean():.3f} ± {scores.std():.3f}")
    cv_results.append([name, scores.mean(), scores.std()])

cv_df = pd.DataFrame(cv_results, columns=["Model", "CV_R2_Mean", "CV_R2_STD"])
print(cv_df)
'''

df['volume_level'] = pd.qcut(df['traffic_volume'], q=3, labels=['Low', 'Medium', 'High'])
le = LabelEncoder()
df['volume_level_encoded'] = le.fit_transform(df['volume_level'])


features = ['hour_sin', 'hour_cos', 'weekday', 'is_holiday', 'is_weekend']
target = 'volume_level_encoded'

train_size = int(len(df) * 0.8)
X_train = df[features].iloc[:train_size]
y_train = df[target].iloc[:train_size]
X_test = df[features].iloc[train_size:]
y_test = df[target].iloc[train_size:]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

results = []
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append([name, acc, prec, rec, f1, cv_scores.mean(), cv_scores.std()])

    print(f"\n{name}:")
    print(f"  Accuracy = {acc:.3f}")
    print(f"  Precision = {prec:.3f}")
    print(f"  Recall = {rec:.3f}")
    print(f"  F1-score = {f1:.3f}")
    print(f"  CV Accuracy Mean = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'CV_Mean', 'CV_Std'])
    print("\nClassification Results Summary:")
    print(results_df)
