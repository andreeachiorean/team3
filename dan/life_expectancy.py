import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np



data = pd.read_csv(r"C:\Users\daxen\Desktop\suport curs\team3\data\Life_Expectancy_Data_new.csv")
data.columns = data.columns.str.strip()
missing = data.isnull().sum()
print(missing[missing > 0])

df = pd.DataFrame(data)
exclude = ['country', 'year', 'status']
cols_to_check = [c for c in df.columns if c.lower() not in exclude]
rezultat = []

for country, group in df.groupby('Country'):
    for col in cols_to_check:
        if group[col].isnull().all():
            rezultat.append({'Country': country, 'Empty_Column': col})
print("--------detalii date lipsa--------")
if rezultat:
    summary_df = pd.DataFrame(rezultat).drop_duplicates()
    summary_df = summary_df.sort_values(by=['Country', 'Empty_Column'])
    print(f"sunt gasite {len(summary_df)} countries cu date complet lipsa pentru toti anii:")
    print(summary_df.to_string(index=False))
    print("\n--- countries unice afectate ---")
    unique_countries = summary_df['Country'].unique()
    unique_countries.sort()
    print(f"Total countries cu date lipsa complet pentru toti anii pentru anumite coloane: {len(unique_countries)}")
    print("Lista countries:")
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
