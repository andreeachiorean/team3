import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import ReLU
from sklearn.preprocessing import StandardScaler
from keras import Sequential, Input
from keras.layers import Dense, LeakyReLU
from keras.layers import Dropout
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)


data = pd.read_csv("..\data\Life_Expectancy_Data_new.csv")
data.columns = data.columns.str.strip()

df = pd.DataFrame(data)
exclude = ['Country', 'Year', 'Status']

cols_to_check = []
for c in df.columns:
    if c not in exclude:
        cols_to_check.append(c)

rezultat = []
for country, group in df.groupby('Country'):
    for col in cols_to_check:
        if group[col].isnull().all():
            rezultat.append({'Country': country, 'Empty_Column': col})

print("--------detalii date lipsa--------")

if rezultat:
    summary_df = pd.DataFrame(rezultat).drop_duplicates()
    summary_df = summary_df.sort_values(by=['Country', 'Empty_Column'])
    print(f"sunt gasite {len(summary_df)} combinatii 'countries-empty_col' cu date complet lipsa pentru toti anii:")
    print(summary_df.to_string(index=False))
    print("\n----------------countries unice afectate -----------------")
    unique_countries = list(summary_df['Country'].unique())
    unique_countries.sort()
    print(f"Numar countries cu date lipsa complet pentru toti anii pentru anumite coloane: {len(unique_countries)}. Acestea sunt:")
    for country in unique_countries:
        print(f"- {country}")
no_data_countries = [item['Country'] for item in rezultat]
no_data_countries_unique = list(set(no_data_countries))

df_cleaned = df[~df['Country'].isin(no_data_countries_unique)].copy()
print("\n-------------data cleaned info()---------------")
df_cleaned.info()
print("\n---------rezulate curatare data frame---------")
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

df_cor = [e for e in df_cleaned.columns if e not in ['Country', 'Year', 'Status']]
# Calculam matricea de corelatie
correlation_matrix = df_cleaned[df_cor].corr()

#Graficul matricei de corelatie
# plt.figure(figsize=(16,12))
# sns.heatmap(correlation_matrix,annot=True, fmt=".2f", cmap="coolwarm",
#             center=0, square = True, linewidths =0.5, cbar_kws={"shrink": .8})
# plt.title("Matricea de corelatie", fontsize = 16, fontweight = 'bold')
# plt.tight_layout()
# plt.show()
#considerat leakage media pe tot datasetul
#df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True)) # inlocuim valorile nule cu media cloanei, pentru a putea rula modeul cu standardscaler

output_col = 'Life expectancy'
input_col = [coloana for coloana in df_cleaned.columns if coloana not in [output_col,'Country', 'Year', 'Status']]
X = df_cleaned[input_col].values
y = df_cleaned[output_col].values

#impartit in train, validation si test
years = sorted(df_cleaned['Year'].unique())

train_years = years[:10]      # 2000–2009
val_years   = years[10:12]    # 2010–2011
test_years  = years[12:]      # 2012–2015

train_df = df_cleaned[df_cleaned['Year'].isin(train_years)]
val_df   = df_cleaned[df_cleaned['Year'].isin(val_years)]
test_df  = df_cleaned[df_cleaned['Year'].isin(test_years)]

#calculez mediile DOAR pe train și le aplic la val/test:
train_means = train_df.mean(numeric_only=True)

train_df = train_df.fillna(train_means)
val_df   = val_df.fillna(train_means)
test_df  = test_df.fillna(train_means)

# ------------------------
# Features / target
# ------------------------
X_train = train_df[input_col].values
y_train = train_df[output_col].values

X_val = val_df[input_col].values
y_val = val_df[output_col].values

X_test = test_df[input_col].values
y_test = test_df[output_col].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
#xgboost sau gradient boost ca modele
#RELU
# dropout_rate = 0.1  # 0.0–0.4
# model = Sequential([
#     Input(shape=(X_train_scaled.shape[1],)),
#     Dense(128), ReLU(),
#     Dropout(dropout_rate),
#     Dense(64), ReLU(),
#     Dropout(dropout_rate),
#     Dense(32), ReLU(),
#     Dropout(dropout_rate),
#     Dense(1, activation='linear')
# ])

#De ce 128–64–16 e mai potrivit cu LeakyReLU
# LeakyReLU deja crește capacitatea efectivă
# Spre deosebire de ReLU:
# LeakyReLU nu „omoară” neuroni
# permite propagarea gradientului și pe valori negative
#  Asta face ca fiecare neuron să fie mai „puternic”.
# Deci nu ai nevoie de la fel de mulți neuroni ca la ReLU.
# αlpha spune cât de mult din valoarea negativă este lăsată să treacă.
# αlpha = 0 → ReLU clasic (tot ce e negativ devine 0)
# αlpha = 0.01 → 1% din valoarea negativă trece mai departe
# αlpha = 0.1 → 10% din valoarea negativă trece mai departe
# model = Sequential([
#     Input(shape=(X_train_scaled.shape[1],)),
#     Dense(128), LeakyReLU(alpha=0.01),
#     Dense(64), LeakyReLU(alpha=0.01),
#     Dense(16), LeakyReLU(alpha=0.01),
#     Dense(1, activation='linear')
# ])

dropout_rate = 0.1

model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128), LeakyReLU(alpha=0.01),
    Dropout(dropout_rate),
    Dense(64), LeakyReLU(alpha=0.01),
    Dropout(dropout_rate),
    Dense(16), LeakyReLU(alpha=0.01),
    Dropout(dropout_rate),
    Dense(1, activation='linear')
])


model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)


early_stop = EarlyStopping(
    monitor="val_mae",       # sau "val_loss"
    patience=8,              # cate epoci astepti fara imbunatatire
    min_delta=0.001,         # imbunatatire minima considerata relevanta
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_mae",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],#  reduce_lr],
    verbose=1
)

# plt.figure()
#
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Train Loss (MSE)')
# plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
# plt.title('Model Loss Progression')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='Train MAE')
# plt.plot(history.history['val_mae'], label='Val MAE')
# plt.title('Mean Absolute Error (Years)')
# plt.xlabel('Epoch')
# plt.ylabel('Error in Years')
# plt.legend()
# plt.show()


predictions = model.predict(X_test_scaled).flatten()

#========

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

# --- erori element-wise ---
errors = predictions - y_test
abs_errors = np.abs(errors)

# --- metrici principale ---
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
medae = median_absolute_error(y_test, predictions)

# Adjusted R^2
n = len(y_test)
p = X_test.shape[1]  # nr. feature-uri
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# % in intervale
within_1y = np.mean(abs_errors <= 1.0) * 100
within_2y = np.mean(abs_errors <= 2.0) * 100

# Max Error + tara/an
idx_max = int(np.argmax(abs_errors))
max_err = abs_errors[idx_max]

max_country = test_df.iloc[idx_max]["Country"]
max_year = test_df.iloc[idx_max]["Year"]
max_actual = y_test[idx_max]
max_pred = predictions[idx_max]
max_signed_err = errors[idx_max]  # semn: + suprapredictie, - subpredictie

print(f"\n----------------Performanta Model (TEST)---------------------")
print("Prezicere speranta de viata:")
print(f"\nMAE: {mae:.2f} years")
##În medie, greșesc cu ~X ani, dar erorile mari sunt penalizate
print("\nRMSE - RMSE penalizează mai sever erorile mari, astfel evidențiază instabilitățile modelului și cazurile extreme, spre deosebire de MAE care tratează toate erorile egal")
print(f"RMSE: {rmse:.2f} years")
print("\nR2: ce procent din variația speranței de viață este explicat de model")
print(f"R2: {r2:.3f}")

#Adjusted R² este o variantă corectată a lui R² care:
# ține cont de numărul de feature-uri
# penalizează modelele care adaugă variabile fără să aducă informație reală
print("\nAdjusted R2 - Cât de bine explică modelul variabila țintă, raportat la complexitatea lui, Adjusted R² < R², înseamnă că unele feature-uri nu ajută")
print(f"Adjusted R2: {adj_r2:.3f}")

# Mediana erorilor absolute:
# calculează eroarea pentru fiecare predicție
# ia valoarea din mijloc, nu media
# De ce e importantă
# Este robustă la outliers.
# Exemplu:
# MAE = 2.5 ani
# Median AE = 1.3 ani
# Înseamnă că:
# majoritatea predicțiilor sunt bune
#
# dar câteva cazuri extreme trag MAE în sus
print("\nMedian AE -  Pentru 50% din observații, greșeala este sub X ani.- eroarea tipică pentru majoritatea observațiilor")
print(f"Median AE: {medae:.2f} years")
print("\nProcentul de predicții care au o eroare mai mică sau egală cu 1 an.")
print(f"Within ±1 year: {within_1y:.1f}%")
print("\nProcentul de predicții cu eroare ≤ 2 ani.")
print(f"Within ±2 years: {within_2y:.1f}%")
print(f"\nMax Error (abs): {max_err:.2f} years")
print(f"    Country: {max_country} | Year: {max_year}")
print(f"    Actual: {max_actual:.2f} | Predicted: {max_pred:.2f} | Signed error: {max_signed_err:+.2f}")

#==========


countries_test = test_df['Country'].values
year_test = test_df['Year'].values
compare = pd.DataFrame({
    'Country': countries_test,
    'Year': year_test,
    'Actual': y_test,
    'Predicted': predictions
})

print("\nSample de 30 predictii:")
print(compare.head(30))



# print("-------------------------------------")
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
#
# x = df['BMI']
# y = df['under-five deaths']
# z = df['Measles']
# def normalize(col):
#     return (col - col.min()) / (col.max() - col.min())
# colors = list(zip(normalize(x),
#                   normalize(y),
#                   normalize(z)))
#
# scatter = ax.scatter(x,y,z, c=colors)
#
# fig.colorbar(scatter, ax=ax, label='Life Expectancy')
# plt.savefig('Life_expectancy.png')
# plt.show()
#
# numeric_cols = df.select_dtypes(include=[np.number]).columns
#
# n_cols = 3
# n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
#
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
# axes = axes.flatten()
#
# for i, col in enumerate(numeric_cols):
#     sns.boxplot(x=df[col], ax=axes[i], color='lightgreen', fliersize=5)
#     axes[i].set_title(f'Distribution of {col}', fontsize=8, fontweight='bold')
#     axes[i].set_xlabel('', fontsize=8)  # Clear x-label for a cleaner look
#     axes[i].grid(axis='x', linestyle='--', alpha=0.5)
#
#
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout()
# plt.savefig('Boxplot.png')
# plt.show()
#
#
