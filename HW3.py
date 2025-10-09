# HW3.py  — Boston Housing: OLS / Ridge / LASSO (train/test + plots on screen)
# 用法：把官方檔 boston.txt 放在同一資料夾後執行
import re, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error

PATH_TXT = "boston.txt"

def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def report(y_tr, yh_tr, y_te, yh_te, name):
    r = dict(
        train_R2=r2_score(y_tr, yh_tr),
        train_RMSE=rmse_np(y_tr, yh_tr),
        train_MAE=mean_absolute_error(y_tr, yh_tr),
        test_R2=r2_score(y_te, yh_te),
        test_RMSE=rmse_np(y_te, yh_te),
        test_MAE=mean_absolute_error(y_te, yh_te),
        model=name,
    )
    print(f"{name} | Train R2={r['train_R2']:.3f}, RMSE={r['train_RMSE']:.3f}, "
          f"Test R2={r['test_R2']:.3f}, RMSE={r['test_RMSE']:.3f}")
    return r

def load_boston_txt(path=PATH_TXT):
    assert os.path.exists(path), "請把 http://lib.stat.cmu.edu/datasets/boston 下載為 boston.txt 放同目錄"
    raw = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    start = next(i for i, ln in enumerate(raw) if re.match(r"^\s*-?\d", ln))
    lines = raw[start:]
    rows = []
    for i in range(0, len(lines), 2):
        if i+1 >= len(lines): break
        rows.append(lines[i].split() + lines[i+1].split())
    cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    df = pd.DataFrame(rows, columns=cols).astype(float)
    df["CHAS"] = df["CHAS"].astype(int); df["RAD"] = df["RAD"].astype(int)
    return df

def main():
    df = load_boston_txt()
    print(df.head())

    X = df.drop(columns=["MEDV"]); y = df["MEDV"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # (a) OLS subset
    subset = ["RM","LSTAT","PTRATIO","NOX","DIS","CHAS"]
    ols = LinearRegression().fit(X_tr[subset], y_tr)
    yh_tr_ols = ols.predict(X_tr[subset]); yh_te_ols = ols.predict(X_te[subset])
    print("OLS coefficients:", pd.Series(ols.coef_, index=subset).round(3).to_dict())
    res_ols = report(y_tr, yh_tr_ols, y_te, yh_te_ols, "OLS (subset)")

    # (b)(c) Ridge / LASSO with scaling
    sc = StandardScaler(); Xtr_sc = sc.fit_transform(X_tr); Xte_sc = sc.transform(X_te)

    ridge = RidgeCV(alphas=np.logspace(-3,3,50), cv=10).fit(Xtr_sc, y_tr)
    yh_tr_r = ridge.predict(Xtr_sc); yh_te_r = ridge.predict(Xte_sc)
    res_ridge = report(y_tr, yh_tr_r, y_te, yh_te_r, f"Ridge (alpha*={ridge.alpha_:.3g})")

    lasso = LassoCV(cv=10, random_state=42, max_iter=10000).fit(Xtr_sc, y_tr)
    yh_tr_l = lasso.predict(Xtr_sc); yh_te_l = lasso.predict(Xte_sc)
    res_lasso = report(y_tr, yh_tr_l, y_te, yh_te_l, f"LASSO (alpha*={lasso.alpha_:.3g})")
    nz = (pd.Series(lasso.coef_, index=X.columns)!=0).sum()
    print(f"LASSO non-zero features: {nz}/13")

    # === Minimal plots (show only, not saving) ===
    # 1) MEDV histogram
    plt.figure(); plt.hist(df["MEDV"], bins=30); plt.xlabel("MEDV ($1000s)"); plt.ylabel("Count")
    plt.title("Distribution of MEDV"); plt.show()

    # 2) RM vs MEDV
   
