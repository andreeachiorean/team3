import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# Changed import to avoid keras.src internal reference errors
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential, Input
from keras.layers import Dense
from keras.utils import to_categorical

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

df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True)) # inlocuim valorile nule cu media cloanei, pentru a putea rula modeul

output_col = 'Life expectancy'
input_col = [coloana for coloana in df_cleaned.columns if coloana not in [output_col,'Country', 'Year', 'Status']]
X = df_cleaned[input_col].values
y = df_cleaned[output_col].values

years = sorted(df_cleaned['Year'].unique())
train_years = years[:12]
test_years = years[12:]

train_df = df_cleaned[df_cleaned['Year'].isin(train_years)]
test_df = df_cleaned[df_cleaned['Year'].isin(test_years)]

X_train = train_df[input_col].values
y_train = train_df[output_col].values
X_test = test_df[input_col].values
y_test = test_df[output_col].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1,activation='linear')
])

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)


history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
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

print(f"\n----------------Performanta Model---------------------")
print(f"Mean Absolute Error: {mae:.2f} years")
print(f"R2 Score: {r2:.2f}")

countries_test = test_df['Country'].values
year_test = test_df['Year'].values
compare = pd.DataFrame({
    'Country': countries_test,
    'Year': year_test,
    'Actual': y_test,
    'Predicted': predictions
})

print("\nSample de predictii:")
print(compare.head(15))



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
