import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from keras.src.layers import Dropout
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras import Sequential, Input
from keras.layers import Dense, LeakyReLU
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



data = pd.read_csv(r"C:\Users\daxen\Desktop\suport curs\team3\data\Life_Expectancy_Data_new.csv")
data.columns = data.columns.str.strip()

df = pd.DataFrame(data)
exclude = ['Country', 'Year', 'Status']
cols_to_check = [c for c in df.columns if c not in exclude]

rezultat = []
for country, group in df.groupby('Country'):
    for col in cols_to_check:
        if group[col].isnull().all():
            rezultat.append({'Country': country, 'Empty_Column': col})

print("--------detalii date lipsa--------")
if rezultat:
    summary_df = pd.DataFrame(rezultat).drop_duplicates()
    summary_df = summary_df.sort_values(by=['Country', 'Empty_Column'])
    print(f"sunt gasite {len(summary_df)} combinatii 'countries-empty_col':")
    print(summary_df.to_string(index=False))

no_data_countries_unique = list(set([item['Country'] for item in rezultat]))
df_cleaned = df[~df['Country'].isin(no_data_countries_unique)].copy()

# --- 2. MATRICEA DE CORELAȚIE ---
df_cor = [e for e in df_cleaned.columns if e not in ['Country', 'Year', 'Status']]
correlation_matrix = df_cleaned[df_cor].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Matricea de corelatie", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))

# -------configurarea modelului--------
output_col = 'Life expectancy'
years = sorted(df_cleaned['Year'].unique())
train_years = years[:12]
test_years = years[12:]


def build_model(input_shape):
    return Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        Dense(16),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='linear')
    ])


# EXPERIMENT 1: MODEL FĂRĂ CLUSTERING (BASELINE)
# =======================================================
print("\n>>> Rulare Model 1: Fara Clustering (Baseline)...")
input_col_1 = [c for c in df_cleaned.columns if c not in [output_col, 'Country', 'Year', 'Status']]

train_df_1 = df_cleaned[df_cleaned['Year'].isin(train_years)]
test_df_1 = df_cleaned[df_cleaned['Year'].isin(test_years)]

X_train_1 = train_df_1[input_col_1].values
y_train_1 = train_df_1[output_col].values
X_test_1 = test_df_1[input_col_1].values
y_test_1 = test_df_1[output_col].values

scaler1 = StandardScaler()
X_train_1_s = scaler1.fit_transform(X_train_1)
X_test_1_s = scaler1.transform(X_test_1)

model1 = build_model(X_train_1_s.shape[1])
model1.compile(loss='mse', optimizer='adam', metrics=['mae'])
model1.fit(X_train_1_s, y_train_1, epochs=40, batch_size=32, verbose=0)

pred1 = model1.predict(X_test_1_s).flatten()
mae_baseline = mean_absolute_error(y_test_1, pred1)
r2_baseline = r2_score(y_test_1, pred1)


#EXPERIMENT 2: OPTIMIZARE NUMAR DE CLUSTERE (gasire numar optim de clustere)
#==========================================================================

print("\n>>> Incepere simulari pentru numarul de clustere...")
df_feat = df_cleaned.copy()
cols_economy = ['GDP_new', 'percentage expenditure', 'Total expenditure']
cols_health = ['BMI', 'under-five deaths', 'Hepatitis B', 'Diphtheria', 'HIV/AIDS', 'Polio']
cols_school = ['Schooling', 'Income composition of resources']

scaler_pca = StandardScaler()
df_feat['dim_economy'] = PCA(n_components=1).fit_transform(scaler_pca.fit_transform(df_feat[cols_economy]))
df_feat['dim_health'] = PCA(n_components=1).fit_transform(scaler_pca.fit_transform(df_feat[cols_health]))
df_feat['dim_schooling'] = PCA(n_components=1).fit_transform(scaler_pca.fit_transform(df_feat[cols_school]))

sim_results = []
cluster_range = range(2, 11)  # Testam de la 2 la 10 clustere

for k in cluster_range:
    X_3d = df_feat[['dim_economy', 'dim_health', 'dim_schooling']].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_feat['Cluster_ID'] = kmeans.fit_predict(X_3d)

    input_col_2 = [c for c in df_feat.columns if c not in [output_col, 'Country', 'Year', 'Status']]
    train_df_2 = df_feat[df_feat['Year'].isin(train_years)]
    test_df_2 = df_feat[df_feat['Year'].isin(test_years)]

    X_train_2 = train_df_2[input_col_2].values
    y_train_2 = train_df_2[output_col].values
    X_test_2 = test_df_2[input_col_2].values
    y_test_2 = test_df_2[output_col].values

    scaler2 = StandardScaler()
    X_train_2_s = scaler2.fit_transform(X_train_2)
    X_test_2_s = scaler2.transform(X_test_2)

    model_sim = build_model(X_train_2_s.shape[1])
    model_sim.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model_sim.fit(X_train_2_s, y_train_2, epochs=40, batch_size=32, verbose=0)

    pred_sim = model_sim.predict(X_test_2_s).flatten()
    mae_k = mean_absolute_error(y_test_2, pred_sim)
    r2_k = r2_score(y_test_2, pred_sim)

    # Verificam daca ambele conditii de imbunatatire sunt indeplinite
    improved = "DA" if (mae_k < mae_baseline and r2_k > r2_baseline) else "NU"
    sim_results.append({'k': k, 'MAE': mae_k, 'R2': r2_k, 'Improved': improved})
    print(f"Clusters: {k} | MAE: {mae_k:.4f} | R2: {r2_k:.4f} | Imbunatatit: {improved}")

# --------------rezultate optimizare---------------------------
sim_df = pd.DataFrame(sim_results)

# Gasim k minim care imbunatateste ambele metrice
optimal_k_row = sim_df[sim_df['Improved'] == 'DA'].first_valid_index()

if optimal_k_row is not None:
    best_k = sim_df.loc[optimal_k_row, 'k']
    print(f"\nNUMARUL MINIM DE CLUSTERE PENTRU IMBUNATATIRE ESTE: {best_k}")
else:
    best_k = 3  # revenire la valoarea ta initiala daca nicio simulare nu a batut modelul simplu
    print("\n!!! Nicio configuratie nu a imbunatatit ambele metrice simultan in aceasta rulare.")

# Grafic mae si mse
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(sim_df['k'], sim_df['MAE'], marker='o', label='MAE Model Clustering')
plt.axhline(y=mae_baseline, color='r', linestyle='--', label='MAE Baseline')
plt.title('MAE vs Numar Clustere')
plt.xlabel('K (Nr de Clustere)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sim_df['k'], sim_df['R2'], marker='o', color='green', label='R2 Model Clustering')
plt.axhline(y=r2_baseline, color='r', linestyle='--', label='R2 Baseline')
plt.title('R2 vs Numar Clustere')
plt.xlabel('K (Nr de Clustere)')
plt.legend()
plt.tight_layout()
plt.show()

# -------------rerulare model pt numar de clustere K optim----------------- ---
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_feat['Cluster_ID'] = kmeans_final.fit_predict(df_feat[['dim_economy', 'dim_health', 'dim_schooling']].values)

# ---grafic 3d clusetere--
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_feat['Cluster_ID'] = kmeans_final.fit_predict(df_feat[['dim_economy', 'dim_health', 'dim_schooling']].values)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_feat['dim_economy'],
                     df_feat['dim_health'],
                     df_feat['dim_schooling'],
                     c=df_feat['Cluster_ID'],
                     cmap='viridis', s=40, alpha=0.7)
ax.set_xlabel('Dimensiune Economie (PCA 1)', fontsize=10, labelpad=10)
ax.set_ylabel('Dimensiune Sănătate (PCA 1)', fontsize=10, labelpad=10)
ax.set_zlabel('Dimensiune Școlarizare (PCA 1)', fontsize=10, labelpad=10)

plt.title(f'Clustering best k={best_k} - Vizualizare 3D', fontsize=14)
fig.colorbar(scatter, ax=ax, label='ID Cluster', shrink=0.5)
plt.show()

# ---------------------tabel cu tarile pe numarul optim de clustere ----------------
cluster_mapping = df_feat.groupby('Country')['Cluster_ID'].first().reset_index()
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
table_data = []
for i in range(best_k):
    countries = cluster_mapping[cluster_mapping['Cluster_ID'] == i]['Country'].tolist()
    chunked_countries = ", ".join(countries[:12]) + ("..." if len(countries) > 12 else "")
    table_data.append([f"Cluster {i}", len(countries), chunked_countries])

table = ax.table(cellText=table_data,
                 colLabels=["ID Cluster", "Nr. Tari", "Exemple Tari"],
                 loc='center',
                 cellLoc='left',
                 colWidths=[0.1, 0.1, 0.8])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.8)
plt.title(f"Repartizarea Tarilor pe cele {best_k} Clustere", fontsize=14, pad=20)
plt.show()

for i in range(best_k):
    print(f"\nTari in Cluster {i}:")
    print(", ".join(cluster_mapping[cluster_mapping['Cluster_ID'] == i]['Country'].tolist()))