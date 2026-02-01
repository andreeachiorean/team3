import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np



df = pd.read_csv(r"C:\Users\daxen\PycharmProjects\team3\data\Life Expectancy Data.xls")
print(df.sample(5))
df.columns = df.columns.str.strip()

missing = df.isnull().sum()
print(missing[missing > 0])

# 3. Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define your axes
x = df['BMI']
y = df['under-five deaths']
z = df['Measles']
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())
colors = list(zip(normalize(x),
                  normalize(y),
                  normalize(z)))

scatter = ax.scatter(x,y,z, c=colors)

fig.colorbar(scatter, ax=ax, label='Life Expectancy')
plt.savefig('Life_expectancy.png')
plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns
# 3. Define the grid size (e.g., 3 columns per row)
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

# 4. Create the subplots figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten() # Flatten to 1D array for easier looping

# 5. Loop through columns and create a boxplot for each
for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df[col], ax=axes[i], color='lightgreen', fliersize=5)
    axes[i].set_title(f'Distribution of {col}', fontsize=8, fontweight='bold')
    axes[i].set_xlabel('', fontsize=8)  # Clear x-label for a cleaner look
    axes[i].grid(axis='x', linestyle='--', alpha=0.5)

# 6. Remove any empty subplots (if the grid is larger than the number of columns)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('Boxplot.png')
plt.show()


