import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

path = r"C:\Users\daxen\Desktop\suport curs\team3\data\Life_Expectancy_Data_new.csv"
data = pd.read_csv(path)
data.columns = data.columns.str.strip()

df_cleaned = data.dropna(subset=['Life expectancy']).copy()
df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))

cols_economy = ['GDP_new', 'percentage expenditure', 'Total expenditure']
cols_health = ['BMI', 'under-five deaths', 'Hepatitis B', 'Diphtheria', 'HIV/AIDS', 'Polio']
cols_school = ['Schooling', 'Income composition of resources']

scaler_pca = StandardScaler()
df_feat = df_cleaned.copy()
df_feat['dim_economy'] = PCA(n_components=1).fit_transform(scaler_pca.fit_transform(df_feat[cols_economy]))
df_feat['dim_health'] = PCA(n_components=1).fit_transform(scaler_pca.fit_transform(df_feat[cols_health]))
df_feat['dim_schooling'] = PCA(n_components=1).fit_transform(scaler_pca.fit_transform(df_feat[cols_school]))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_feat['Cluster_ID'] = kmeans.fit_predict(df_feat[['dim_economy', 'dim_health', 'dim_schooling']])

cluster_stats = df_feat.groupby('Cluster_ID')['Life expectancy'].mean().sort_values()
color_map_dict = {cluster_stats.index[0]: 'red', cluster_stats.index[1]: 'yellow', cluster_stats.index[2]: 'green'}
ordered_colors = [color_map_dict[i] for i in range(3)]
custom_cmap = ListedColormap(ordered_colors)
tari_cluster = df_feat.groupby('Country')['Cluster_ID'].first().reset_index()

# ------vizualizare 3 clustere 3D ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_feat['dim_economy'], df_feat['dim_health'], df_feat['dim_schooling'],
                     c=df_feat['Cluster_ID'], cmap=custom_cmap, s=50, alpha=0.7, edgecolors='w')
ax.set_xlabel('Dimensiune Economie (PCA 1)', fontsize=10, labelpad=10)
ax.set_ylabel('Dimensiune Sanatate (PCA 1)', fontsize=10, labelpad=10)
ax.set_zlabel('Dimensiune Scolarizare (PCA 1)', fontsize=10, labelpad=10)
ax.set_title('Vizualizare 3D: Cele 3 Clustere (Economie, Sanatate, Scoala)')
plt.show()
#--------------corectii nume tari sa nu apara gri pe harta ---------------
nume_corectate = {
    'United States of America': 'United States',
    'Russian Federation': 'Russia',
    'Viet Nam': 'Vietnam',
    'Republic of Moldova': 'Moldova',
    'Iran (Islamic Republic of)': 'Iran',
    'Syrian Arab Republic': 'Syria',
    'Republic of Korea': 'South Korea',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom'
}
tari_cluster['Country'] = tari_cluster['Country'].replace(nume_corectate)

print("\n>> Generare heatmap harta lumii pe clustere...")

world_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
world = gpd.read_file(world_url)
world = world.rename(columns={'name': 'Country'})
world_map = world.merge(tari_cluster, on='Country', how='left')

# ------calculare centroizi pentru precizie label nume tari pe harta -----------------------
world_map['centroid'] = world_map.to_crs(epsg=3857).geometry.centroid.to_crs(world_map.crs)

fig, ax = plt.subplots(1, 1, figsize=(25, 15))
world.plot(ax=ax, color='#f2f2f2', edgecolor='#bcbcbc', linewidth=0.5)
world_map.dropna(subset=['Cluster_ID']).plot(
    column='Cluster_ID',
    ax=ax,
    cmap=custom_cmap,
    edgecolor='black',
    linewidth=0.3
)

for idx, row in world_map.iterrows():
    if not pd.isna(row['Cluster_ID']):
        if row.geometry.area > 10: #nu supraincarcam harta cu nume tari, pt tarile cu suprafata mica
            ax.text(row['centroid'].x, row['centroid'].y,
                    s=row['Country'],
                    fontsize=7,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='black',
                    alpha=0.7)

legend_elements = [
    Line2D([0], [0], marker='s', color='w', label='Speranta de viata: SCAZUTA', markerfacecolor='red', markersize=12),
    Line2D([0], [0], marker='s', color='w', label='Speranta de viata: MEDIE', markerfacecolor='yellow', markersize=12),
    Line2D([0], [0], marker='s', color='w', label='Speranta de viata: RIDICATA', markerfacecolor='green', markersize=12)
]
ax.legend(handles=legend_elements, loc='lower left', title="Clustere")

plt.title("Harta Lumii: Analiza Sperantei de Viata pe 3 Clustere", fontsize=20, pad=20)
plt.axis('off')
plt.savefig('Geomap_Life_expectancy.png', dpi=300, bbox_inches='tight')
plt.show()