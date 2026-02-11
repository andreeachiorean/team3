import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

# ======================================================
# 0) CONFIG
#
# Best params:
#   objective: reg:squarederror
#   eval_metric: mae
#   seed: 42
#   subsample: 0.6
#   min_child_weight: 2
#   max_depth: 6
#   lambda: 1.0
#   gamma: 0.0
#   eta: 0.05
#   colsample_bytree: 0.6
#   alpha: 0.0001
# ======================================================
CONFIG = {
    "random_state": 42,
    "data_path": "../data/Life_Expectancy_Data_new.csv",

    # Split temporal (index în lista de ani sortați)
    "split": {
        "train_years_count": 10,   # primii 10 ani 2000-2009
        "val_years_count": 2,      # următorii 2 ani 2010-2011
        # restul = test
    },

    # Train control
    "train": {
        "max_rounds": 5000,
        "early_stopping_rounds": 50,
        "verbose_eval": 50,  # la câte runde afișează progres; pune False dacă vrei silent
    },

    # XGBoost params (train API)
    "xgb_params": {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "gamma": 0.0,
        "alpha": 0.0001,
        "lambda": 1.0,
    },

    # Output
    "output_dir": "artifacts",
}


# ======================================================
# 0.1) LOGGING cu timestamp (console + file)
# ======================================================
def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"train_{ts}.log")

    logger = logging.getLogger("xgb_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # evită dublarea handlerelor în rerun-uri

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_path}")
    return logger


# ======================================================
# 1) LOAD + CLEAN DATA
# ======================================================
def load_and_clean(data_path: str, logger: logging.Logger) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    df = data.copy()

    exclude = ["Country", "Year", "Status"]
    cols_to_check = [c for c in df.columns if c not in exclude]

    rezultat = []
    for country, group in df.groupby("Country"):
        for col in cols_to_check:
            if group[col].isnull().all():
                rezultat.append({"Country": country, "Empty_Column": col})

    logger.info("-------- detalii date lipsa --------")
    if rezultat:
        summary_df = (
            pd.DataFrame(rezultat)
            .drop_duplicates()
            .sort_values(by=["Country", "Empty_Column"])
        )
        logger.info(
            f"Sunt gasite {len(summary_df)} combinatii 'countries-empty_col' cu date complet lipsa:"
        )
        logger.info("\n" + summary_df.to_string(index=False))

        unique_countries = sorted(summary_df["Country"].unique().tolist())
        logger.info("---------------- countries unice afectate ----------------")
        logger.info(f"Numar countries afectate: {len(unique_countries)}")
        for c in unique_countries:
            logger.info(f"- {c}")

    bad_countries = sorted(set([item["Country"] for item in rezultat]))
    df_cleaned = df[~df["Country"].isin(bad_countries)].copy()

    logger.info("------------- data cleaned info() ---------------")
    buf = []
    df_cleaned.info(buf=buf) if False else None  # păstrat ca idee; info() nu merge direct în list simplu
    logger.info(f"- numar initial de randuri: {len(df)}")
    logger.info(f"- numar randuri sterse: {len(df) - len(df_cleaned)}")
    logger.info(f"- numar de randuri ramase: {len(df_cleaned)}")
    logger.info(f"- numar initial de countries: {df['Country'].nunique()}")
    logger.info(f"- numar countries sterse: {df['Country'].nunique() - df_cleaned['Country'].nunique()}")
    logger.info(f"- numar countries unice ramase: {df_cleaned['Country'].nunique()}")

    # salvăm cleaned data
    cleaned_path = os.path.join(CONFIG["output_dir"], "Cleaned_Life_Expectancy_Data.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned dataset: {cleaned_path}")

    missing = df_cleaned.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        logger.info("--------- coloane ramase cu randuri nule dupa curatare ---------")
        logger.info("\n" + missing.to_string())
    else:
        logger.info("Nu mai exista valori lipsa dupa curatare (inainte de imputare).")

    return df_cleaned


# ======================================================
# 2) FEATURES / TARGET + SPLIT TEMPORAL (fara leakage)
# ======================================================
def temporal_split(df_cleaned: pd.DataFrame, logger: logging.Logger):
    output_col = "Life expectancy"
    input_cols = [c for c in df_cleaned.columns if c not in [output_col, "Country", "Year", "Status"]]

    years = sorted(df_cleaned["Year"].unique())
    n_train = CONFIG["split"]["train_years_count"]
    n_val = CONFIG["split"]["val_years_count"]

    train_years = years[:n_train]
    val_years = years[n_train:n_train + n_val]
    test_years = years[n_train + n_val:]

    logger.info(f"Years (all): {years[0]} ... {years[-1]} | count={len(years)}")
    logger.info(f"Train years: {train_years[0]} ... {train_years[-1]} | count={len(train_years)}")
    logger.info(f"Val years:   {val_years[0]} ... {val_years[-1]} | count={len(val_years)}")
    logger.info(f"Test years:  {test_years[0]} ... {test_years[-1]} | count={len(test_years)}")

    train_df = df_cleaned[df_cleaned["Year"].isin(train_years)].copy().reset_index(drop=True)
    val_df = df_cleaned[df_cleaned["Year"].isin(val_years)].copy().reset_index(drop=True)
    test_df = df_cleaned[df_cleaned["Year"].isin(test_years)].copy().reset_index(drop=True)

    # imputare fara leakage: medii DOAR din train
    train_means = train_df.mean(numeric_only=True)
    train_df = train_df.fillna(train_means)
    val_df = val_df.fillna(train_means)
    test_df = test_df.fillna(train_means)

    X_train = train_df[input_cols].values
    y_train = train_df[output_col].values

    X_val = val_df[input_cols].values
    y_val = val_df[output_col].values

    X_test = test_df[input_cols].values
    y_test = test_df[output_col].values

    return (input_cols, train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test)


# ======================================================
# 3) TRAIN cu EARLY STOPPING + SAVE BEST MODEL
# ======================================================
def train_and_save_best(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    logger: logging.Logger,
    output_dir: str,
):
    params = dict(CONFIG["xgb_params"])
    params["seed"] = CONFIG["random_state"]

    # salvează și config params
    params_path = os.path.join(output_dir, "xgb_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"xgb_params": params, "train": CONFIG["train"], "split": CONFIG["split"]}, f, indent=2)
    logger.info(f"Saved config: {params_path}")

    logger.info("================ TRAIN (early stopping on val) ================")
    evals = [(dtrain, "train"), (dval, "val")]

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=CONFIG["train"]["max_rounds"],
        evals=evals,
        early_stopping_rounds=CONFIG["train"]["early_stopping_rounds"],
        verbose_eval=CONFIG["train"]["verbose_eval"],
    )

    # best_iteration e 0-indexed
    best_iter = int(getattr(bst, "best_iteration", -1))
    best_score = float(getattr(bst, "best_score", np.nan))
    best_num_round = best_iter + 1 if best_iter >= 0 else None

    logger.info("================ BEST (from early stopping) ================")
    logger.info(f"best_score (val mae): {best_score:.6f}")
    logger.info(f"best_iteration: {best_iter} (0-indexed)")
    logger.info(f"best_num_round: {best_num_round}")

    best_model_path = os.path.join(output_dir, "xgb_best_model.json")
    bst.save_model(best_model_path)
    logger.info(f"Saved BEST model: {best_model_path}")

    return bst, best_num_round, best_score, best_model_path


# ======================================================
# 4) FINAL TRAIN pe TRAIN+VAL cu best_num_round + SAVE FINAL
# ======================================================
def train_final_trainval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_num_round: int,
    logger: logging.Logger,
    output_dir: str,
):
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    dtrainval = xgb.DMatrix(X_trainval, label=y_trainval)

    params = dict(CONFIG["xgb_params"])
    params["seed"] = CONFIG["random_state"]

    logger.info("================ FINAL TRAIN (train+val) ================")
    logger.info(f"Training num_boost_round = {best_num_round}")

    final_bst = xgb.train(
        params=params,
        dtrain=dtrainval,
        num_boost_round=best_num_round,
        evals=[],
        verbose_eval=False
    )

    final_model_path = os.path.join(output_dir, "xgb_final_trainval.json")
    final_bst.save_model(final_model_path)
    logger.info(f"Saved FINAL model: {final_model_path}")

    return final_bst, final_model_path


# ======================================================
# 5) EVALUARE + PLOTS
# ======================================================
def evaluate_and_plot(
    model: xgb.Booster,
    dtest: xgb.DMatrix,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    logger: logging.Logger,
    output_dir: str,
    label: str = "XGBoost",
):
    # predict (compatibil)
    try:
        preds = model.predict(dtest)
    except TypeError:
        preds = model.predict(dtest)

    errors = preds - y_test
    abs_errors = np.abs(errors)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    medae = median_absolute_error(y_test, preds)

    n = len(y_test)
    p = dtest.num_col()
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    within_1y = np.mean(abs_errors <= 1.0) * 100
    within_2y = np.mean(abs_errors <= 2.0) * 100

    idx_max = int(np.argmax(abs_errors))
    max_err = float(abs_errors[idx_max])

    max_country = test_df.loc[idx_max, "Country"]
    max_year = test_df.loc[idx_max, "Year"]
    max_actual = float(y_test[idx_max])
    max_pred = float(preds[idx_max])
    max_signed_err = float(errors[idx_max])

    logger.info(f"---------------- Performanta {label} (TEST) ----------------")
    logger.info(f"MAE: {mae:.2f} years")
    logger.info(f"RMSE: {rmse:.2f} years")
    logger.info(f"R2: {r2:.3f}")
    logger.info(f"Adjusted R2: {adj_r2:.3f}")
    logger.info(f"Median AE: {medae:.2f} years")
    logger.info(f"Within ±1 year: {within_1y:.1f}%")
    logger.info(f"Within ±2 years: {within_2y:.1f}%")
    logger.info(f"Max Error (abs): {max_err:.2f} years | {max_country} | {max_year} | "
                f"Actual={max_actual:.2f} Pred={max_pred:.2f} Signed={max_signed_err:+.2f}")

    compare = pd.DataFrame({
        "Country": test_df["Country"].values,
        "Year": test_df["Year"].values,
        "Actual": y_test,
        "Predicted": preds,
        "Error": errors,
        "AbsError": abs_errors
    })

    sample_path = os.path.join(output_dir, "test_predictions_sample.csv")
    compare.head(200).to_csv(sample_path, index=False)
    logger.info(f"Saved prediction sample: {sample_path}")

    # --- Scatter: Predictii vs Reale ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.6)
    mn = float(min(y_test.min(), preds.min()))
    mx = float(max(y_test.max(), preds.max()))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2)
    plt.title(f"{label}: Predictii vs Reale\nR² = {r2:.3f}, MAE = {mae:.2f}")
    plt.xlabel("Valori Reale ('Life Expectancy')")
    plt.ylabel("Predictii")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "xgb_pred_vs_real.png")
    plt.savefig(scatter_path, dpi=200)
    plt.show()
    logger.info(f"Saved plot: {scatter_path}")

    # --- Boxplot: erori ---
    plt.figure(figsize=(7, 6))
    plt.boxplot(errors, labels=[label], showfliers=True)
    plt.axhline(0, linestyle="--", linewidth=2)
    mean_err = float(np.mean(errors))
    plt.text(1.02, mean_err, f"Mean: {mean_err:.2f}", color="red", fontsize=11, fontweight="bold")
    plt.title(f"Distributia erorilor: {label}")
    plt.ylabel("Eroare (Predictie - Real)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    box_path = os.path.join(output_dir, "xgb_error_boxplot.png")
    plt.savefig(box_path, dpi=200)
    plt.show()
    logger.info(f"Saved plot: {box_path}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "medae": medae,
        "within_1y": within_1y,
        "within_2y": within_2y,
        "max_err": max_err,
        "max_country": max_country,
        "max_year": int(max_year),
    }


# ======================================================
# MAIN
# ======================================================
def main():
    np.random.seed(CONFIG["random_state"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger = setup_logger(CONFIG["output_dir"])

    logger.info("========== START ==========")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")

    df_cleaned = load_and_clean(CONFIG["data_path"], logger)

    (
        input_cols,
        train_df, val_df, test_df,
        X_train, y_train, X_val, y_val, X_test, y_test
    ) = temporal_split(df_cleaned, logger)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train best (early stopping on val) + save
    best_bst, best_num_round, best_score, best_model_path = train_and_save_best(
        dtrain=dtrain, dval=dval, logger=logger, output_dir=CONFIG["output_dir"]
    )

    # Optional: final model trainval (recomandat) + save
    if best_num_round is None:
        logger.warning("best_num_round is None (nu am gasit best_iteration). Folosesc max_rounds ca fallback.")
        best_num_round = CONFIG["train"]["max_rounds"]

    final_bst, final_model_path = train_final_trainval(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        best_num_round=best_num_round,
        logger=logger,
        output_dir=CONFIG["output_dir"],
    )

    # Evaluate final on test + plots
    metrics = evaluate_and_plot(
        model=final_bst,
        dtest=dtest,
        test_df=test_df,
        y_test=y_test,
        logger=logger,
        output_dir=CONFIG["output_dir"],
        label="XGBoost (Config + EarlyStopping)",
    )

    metrics_path = os.path.join(CONFIG["output_dir"], "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    logger.info("========== DONE ==========")
    logger.info(f"Best model saved at:  {best_model_path}")
    logger.info(f"Final model saved at: {final_model_path}")


if __name__ == "__main__":
    main()


"""
ex load model
bst = xgb.Booster()
bst.load_model("artifacts/xgb_best_model.json")

dtest = xgb.DMatrix(X_test)
preds = bst.predict(dtest)
"""