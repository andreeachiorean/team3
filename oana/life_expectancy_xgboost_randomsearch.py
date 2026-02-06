import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

# ======================================================
# 0) SETARI
# ======================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "../data/Life_Expectancy_Data_new.csv"

N_TRIALS = 30           # cate combinatii random testezi
MAX_ROUNDS = 5000       # nr maxim de boost rounds (early stopping opreste mai devreme)
EARLY_STOPPING = 50     # cate runde fara imbunatatire astepti

# ======================================================
# 1) LOAD + CLEAN DATA
# ======================================================
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()
df = data.copy()

exclude = ["Country", "Year", "Status"]
cols_to_check = [c for c in df.columns if c not in exclude]

rezultat = []
for country, group in df.groupby("Country"):
    for col in cols_to_check:
        if group[col].isnull().all():
            rezultat.append({"Country": country, "Empty_Column": col})

print("--------detalii date lipsa--------")
if rezultat:
    summary_df = pd.DataFrame(rezultat).drop_duplicates().sort_values(by=["Country", "Empty_Column"])
    print(f"sunt gasite {len(summary_df)} combinatii 'countries-empty_col' cu date complet lipsa pentru toti anii:")
    print(summary_df.to_string(index=False))

    unique_countries = sorted(summary_df["Country"].unique().tolist())
    print("\n----------------countries unice afectate -----------------")
    print(f"Numar countries afectate: {len(unique_countries)}")
    for c in unique_countries:
        print(f"- {c}")

bad_countries = sorted(set([item["Country"] for item in rezultat]))
df_cleaned = df[~df["Country"].isin(bad_countries)].copy()

print("\n-------------data cleaned info()---------------")
df_cleaned.info()

print("\n---------rezultate curatare data frame---------")
print(f"-numar initial de randuri: {len(df)}")
print(f"-numar randuri sterse: {len(df) - len(df_cleaned)}")
print(f"-numar de randuri ramase: {len(df_cleaned)}")
print(f"-numar initial de countries: {df['Country'].nunique()}")
print(f"-numar countries sterse: {df['Country'].nunique() - df_cleaned['Country'].nunique()}")
print(f"-numar countries unice ramase: {df_cleaned['Country'].nunique()}")

df_cleaned.to_csv("Cleaned_Life_Expectancy_Data.csv", index=False)

print("\n---------coloane ramase cu randuri nule dupa curatare data frame---------")
missing = df_cleaned.isnull().sum()
print(missing[missing > 0])

# ======================================================
# 2) FEATURES / TARGET + SPLIT TEMPORAL
# ======================================================
output_col = "Life expectancy"
input_col = [c for c in df_cleaned.columns if c not in [output_col, "Country", "Year", "Status"]]

years = sorted(df_cleaned["Year"].unique())
train_years = years[:10]      # 2000–2009
val_years   = years[10:12]    # 2010–2011
test_years  = years[12:]      # 2012–2015

train_df = df_cleaned[df_cleaned["Year"].isin(train_years)].copy().reset_index(drop=True)
val_df   = df_cleaned[df_cleaned["Year"].isin(val_years)].copy().reset_index(drop=True)
test_df  = df_cleaned[df_cleaned["Year"].isin(test_years)].copy().reset_index(drop=True)

# imputare fara leakage: medii DOAR din train
train_means = train_df.mean(numeric_only=True)
train_df = train_df.fillna(train_means)
val_df   = val_df.fillna(train_means)
test_df  = test_df.fillna(train_means)

X_train = train_df[input_col].values
y_train = train_df[output_col].values

X_val = val_df[input_col].values
y_val = val_df[output_col].values

X_test = test_df[input_col].values
y_test = test_df[output_col].values

# ======================================================
# 3) DMATRIX (format nativ XGBoost)
# ======================================================
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# ======================================================
# 4) RANDOM SEARCH SPACE
#    (in train API, learning_rate = eta, reg_alpha = alpha, reg_lambda = lambda)
# ======================================================
param_dist = {
    "eta": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 2, 3, 5, 8],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0.0, 0.1, 0.2, 0.5, 1.0],
    "alpha": [0.0, 1e-4, 1e-3, 1e-2, 0.1],
    "lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}

samples = list(ParameterSampler(param_dist, n_iter=N_TRIALS, random_state=RANDOM_STATE))

# ======================================================
# 5) RULEAZA RANDOM SEARCH (selectie dupa val MAE)
# ======================================================
best_val_mae = np.inf
best_params = None
best_num_round = None  # num_boost_round

print("\n================ RANDOM SEARCH (XGBoost train API) ================")

for i, p in enumerate(samples, start=1):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "seed": RANDOM_STATE,
        **p
    }

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False
    )

    # best_score/best_iteration sunt disponibile in multe versiuni vechi
    try:
        val_mae = float(bst.best_score)
        best_iter = int(bst.best_iteration)  # 0-indexed
        num_round = best_iter + 1
    except Exception:
        # fallback: luam ultimul scor din evals_result (mai robust)
        evals_result = bst.evals_result()
        mae_list = evals_result["val"]["mae"]
        val_mae = float(mae_list[-1])
        num_round = len(mae_list)

    print(f"[{i:02d}/{N_TRIALS}] val_mae={val_mae:.4f} | num_round={num_round} | params={p}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_params = params
        best_num_round = num_round

print("\n================ BEST TRIAL ================")
print(f"Best val_mae: {best_val_mae:.4f}")
print(f"Best num_boost_round: {best_num_round}")
print("Best params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ======================================================
# 6) ANTRENARE FINALA PE TRAIN+VAL, APOI TEST
# ======================================================
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
dtrainval = xgb.DMatrix(X_trainval, label=y_trainval)

final_bst = xgb.train(
    params=best_params,
    dtrain=dtrainval,
    num_boost_round=best_num_round,
    evals=[],
    verbose_eval=False
)

# Predict pe test (compatibil vechi): ntree_limit poate sa nu existe in unele versiuni
try:
    predictions = final_bst.predict(dtest, ntree_limit=best_num_round)
except TypeError:
    predictions = final_bst.predict(dtest)

# ======================================================
# 7) METRICS PE TEST + MAX ERROR (tara/an)
# ======================================================
errors = predictions - y_test
abs_errors = np.abs(errors)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
medae = median_absolute_error(y_test, predictions)

n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

within_1y = np.mean(abs_errors <= 1.0) * 100
within_2y = np.mean(abs_errors <= 2.0) * 100

idx_max = int(np.argmax(abs_errors))
max_err = abs_errors[idx_max]

max_country = test_df.loc[idx_max, "Country"]
max_year = test_df.loc[idx_max, "Year"]
max_actual = y_test[idx_max]
max_pred = predictions[idx_max]
max_signed_err = errors[idx_max]

print(f"\n----------------Performanta XGBoost RandomSearch (TEST)---------------------")
print(f"MAE: {mae:.2f} years")
print(f"RMSE: {rmse:.2f} years")
print(f"R2: {r2:.3f}")
print(f"Adjusted R2: {adj_r2:.3f}")
print(f"Median AE: {medae:.2f} years")
print(f"Within ±1 year: {within_1y:.1f}%")
print(f"Within ±2 years: {within_2y:.1f}%")

print(f"\nMax Error (abs): {max_err:.2f} years")
print(f"    Country: {max_country} | Year: {max_year}")
print(f"    Actual: {max_actual:.2f} | Predicted: {max_pred:.2f} | Signed error: {max_signed_err:+.2f}")

compare = pd.DataFrame({
    "Country": test_df["Country"].values,
    "Year": test_df["Year"].values,
    "Actual": y_test,
    "Predicted": predictions,
    "Error": errors,
    "AbsError": abs_errors
})

print("\nSample de 30 predictii:")
print(compare.head(30))
