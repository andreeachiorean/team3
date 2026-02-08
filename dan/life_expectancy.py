import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras import Sequential, Input
from keras.layers import Dense, LeakyReLU
import xgboost as xgb

data = pd.read_csv(r"C:\Users\daxen\Desktop\suport curs\team3\data\Life_Expectancy_Data_new.csv")
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
    unique_countries = summary_df['Country'].unique()
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
plt.figure(figsize=(16,12))
sns.heatmap(correlation_matrix,annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square = True, linewidths =0.5, cbar_kws={"shrink": .8})
plt.title("Matricea de corelatie", fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.show()

df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True)) # inlocuim valorile nule cu media cloanei, pentru a putea rula modeul cu standardscaler

output_col = 'Life expectancy'
input_col = [coloana for coloana in df_cleaned.columns if coloana not in [output_col,'Country', 'Year', 'Status']]
X = df_cleaned[input_col].values
y = df_cleaned[output_col].values

years = sorted(df_cleaned['Year'].unique())
train_years = years[:12] #80% din seriile anuale 2000-2015
test_years = years[12:] #20% din seriile anuale 2000-2015

train_df = df_cleaned[df_cleaned['Year'].isin(train_years)]
test_df = df_cleaned[df_cleaned['Year'].isin(test_years)]

X_train = train_df[input_col].values
y_train = train_df[output_col].values
X_test = test_df[input_col].values
y_test = test_df[output_col].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

###------------------XGBOOST------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
#------------------------------------------------------------------------------------------

#------------------Keras sequential--------------
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128),LeakyReLU(alpha=0.01),
    Dense(64),LeakyReLU(alpha=0.01),
    Dense(16),LeakyReLU(alpha=0.01),
    Dense(1, activation='linear')
])

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=40,
    batch_size=32,
    verbose=1
)

plt.figure()

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.title('Model Loss Progression')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Mean Absolute Error (Years)')
plt.xlabel('Epoch')
plt.ylabel('Error in Years')
plt.legend()
plt.show()


predictions = model.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

### --------------XGBOOST: Predictii si Metrici pentru-------------------
predictions_xgb = xgb_model.predict(X_test_scaled)
mae_xgb = mean_absolute_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)
#-----------------------------------------------------------

print(f"\n----------------Performanta Model Keras (DL)---------------------")
print(f"Mean Absolute Error(prezicere speranta de viata cu eroare de): {mae:.2f} years")
print(f"R2 Score: {r2:.2f}")

#-----------XGBOOST: Print rezultate XGBoost ###
print(f"\n----------------Performanta Model XGBoost---------------------")
print(f"Mean Absolute Error: {mae_xgb:.2f} years")
print(f"R2 Score: {r2_xgb:.2f}")
#-----------------------------------------------------------


countries_test = test_df['Country'].values
year_test = test_df['Year'].values
compare = pd.DataFrame({
    'Country': countries_test,
    'Year': year_test,
    'Actual': y_test,
    'Keras_Pred': predictions,
    #  Adaugare XGBOOST in comparatie
    'XGB_Pred': predictions_xgb
})

print("\nSample de 30 predictii (Comparatie Keras vs XGBoost):")
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