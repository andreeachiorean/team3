import numpy as np
import pandas as pd
import keras_tuner as kt
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import Sequential, Input
from keras.layers import Dense, ReLU, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

# ======================================================
# 1) LOAD + CLEAN DATA
# ======================================================
data = pd.read_csv("../data/Life_Expectancy_Data_new.csv")
data.columns = data.columns.str.strip()
df = data.copy()

exclude = ["Country", "Year", "Status"]
cols_to_check = [c for c in df.columns if c not in exclude]

bad_countries = []
for country, group in df.groupby("Country"):
    for col in cols_to_check:
        if group[col].isnull().all():
            bad_countries.append(country)

df = df[~df["Country"].isin(set(bad_countries))].copy()

# ======================================================
# 2) SPLIT TEMPORAL (TRAIN / VAL / TEST)
# ======================================================
years = sorted(df["Year"].unique())
train_years = years[:10]
val_years   = years[10:12]
test_years  = years[12:]

train_df = df[df["Year"].isin(train_years)]
val_df   = df[df["Year"].isin(val_years)]
test_df  = df[df["Year"].isin(test_years)]

# imputare fara leakage
train_means = train_df.mean(numeric_only=True)
train_df = train_df.fillna(train_means)
val_df   = val_df.fillna(train_means)
test_df  = test_df.fillna(train_means)

# ======================================================
# 3) FEATURES / TARGET
# ======================================================
target = "Life expectancy"
features = [c for c in df.columns if c not in ["Country", "Year", "Status", target]]

X_train = train_df[features].values
y_train = train_df[target].values

X_val = val_df[features].values
y_val = val_df[target].values

X_test = test_df[features].values
y_test = test_df[target].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ======================================================
# 4) BUILD MODEL – RANDOM SEARCH + ReLU
# ======================================================
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    # Layer 1
    units1 = hp.Choice("units_1", [64, 128, 256])
    model.add(Dense(units1))
    model.add(ReLU())
    model.add(Dropout(hp.Float("dropout_1", 0.0, 0.4, step=0.1)))

    # Layer 2
    units2 = hp.Choice("units_2", [32, 64, 128])
    model.add(Dense(units2))
    model.add(ReLU())
    model.add(Dropout(hp.Float("dropout_2", 0.0, 0.4, step=0.1)))

    # Bottleneck
    units3 = hp.Choice("units_3", [16, 32])
    model.add(Dense(units3))
    model.add(ReLU())

    model.add(Dense(1, activation="linear"))

    lr = hp.Float("lr", 1e-4, 5e-3, sampling="log")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ======================================================
# 5) RANDOM SEARCH
# ======================================================
tuner = kt.RandomSearch(
    build_model,
    objective="val_mae",
    max_trials=20,
    executions_per_trial=1,
    directory="random_search_relu",
    project_name="life_expectancy",
    overwrite=True
)

early_stop = EarlyStopping(
    monitor="val_mae",
    patience=8,
    restore_best_weights=True
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ======================================================
# 6) BEST MODEL
# ======================================================
best_model = tuner.get_best_models(1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]

print("\nBEST HYPERPARAMETERS:")
for k, v in best_hp.values.items():
    print(f"{k}: {v}")

# ======================================================
# 7) FINAL TEST EVALUATION
# ======================================================
pred = best_model.predict(X_test).flatten()

errors = pred - y_test
abs_errors = np.abs(errors)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
medae = median_absolute_error(y_test, pred)

n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

within_1y = np.mean(abs_errors <= 1) * 100
within_2y = np.mean(abs_errors <= 2) * 100

idx_max = np.argmax(abs_errors)
max_err = abs_errors[idx_max]

print("\n================ TEST PERFORMANCE ================")
print(f"MAE: {mae:.2f} years")
print(f"RMSE: {rmse:.2f} years")
print(f"R2: {r2:.3f}")
print(f"Adjusted R2: {adj_r2:.3f}")
print(f"Median AE: {medae:.2f} years")
print(f"Within ±1 year: {within_1y:.1f}%")
print(f"Within ±2 years: {within_2y:.1f}%")
print(f"Max Error: {max_err:.2f} years")
print(f"Country: {test_df.iloc[idx_max]['Country']} | Year: {test_df.iloc[idx_max]['Year']}")


# =======================
# PLOTS: MLP RandomSearch
# =======================
actual = y_test
preds = pred
err = preds - actual

# --- Scatter: Predictii vs Reale ---
plt.figure(figsize=(8, 6))
plt.scatter(actual, preds, alpha=0.6)
mn = float(min(actual.min(), preds.min()))
mx = float(max(actual.max(), preds.max()))
plt.plot([mn, mx], [mn, mx], "r--", linewidth=2)

plt.title(f"MLP RandomSearch: Predictii vs Reale\nR² = {r2:.3f}, MAE = {mae:.2f}")
plt.xlabel("Valori Reale ('Life Expectancy')")
plt.ylabel("Predictii MLP")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlp_randomsearch_pred_vs_real.png", dpi=200)
plt.show()

# --- Boxplot: erori ---
plt.figure(figsize=(7, 6))
bp = plt.boxplot(err, labels=["MLP RandomSearch"], showfliers=True)

plt.axhline(0, linestyle="--", linewidth=2)
mean_err = float(np.mean(err))
plt.text(
    1.02, mean_err, f"Mean: {mean_err:.2f}",
    color="red", fontsize=11, fontweight="bold"
)

plt.title("Distributia erorilor: MLP RandomSearch")
plt.ylabel("Eroare (Predictie - Real)")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("mlp_randomsearch_error_boxplot.png", dpi=200)
plt.show()
