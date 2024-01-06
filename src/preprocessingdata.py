import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns

df = pd.read_csv('Spotify_Youtube.csv')

print(df.dtypes)
print(df.describe())

# Sample 10% of the data
sampled_data = df.sample(frac=0.1)

df['Danceability'].fillna(df['Danceability'].mean(), inplace=True)

df.drop_duplicates(subset='Track', keep='first', inplace=True)

selected_columns = df[['Track', 'Artist', 'Danceability', 'Energy']]

df['DanceEnergyRatio'] = df['Danceability'] / df['Energy']

# Discretize a column into intervals
bins = [0, 0.5, 1, 2]
df['DanceabilityCategory'] = pd.cut(df['Danceability'], bins)

# Apply standard scaling to a numeric column
scaler = StandardScaler()
df['ScaledEnergy'] = scaler.fit_transform(df[['Energy']])

# Feature Engineering
df['SongLength'] = df['Duration_ms'] / 1000  # Convert milliseconds to seconds

df['HighEnergy'] = df['Energy'].apply(lambda x: 1 if x > 0.8 else 0)  # threshold for high energy: 1 high ; 0 low

df.to_csv('cleaned_data.csv', index=False)

#dimension reduction
# Columns we want to reduct into 2
features = df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Tempo']]

features = features.fillna(features.mean()) # need to fill because NaN is not allowed
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# New DataFrame with data of reduced_features
reduced_df = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])

numerical_columns = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Tempo']

z_scores = df[numerical_columns].apply(zscore)

threshold = 1
outliers = df[(z_scores > threshold).any(axis=1)]

print("Outliers:")
print(outliers)


# Multivariate Analysis (Example: Pairplot)

sns.pairplot(df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Tempo']])
plt.show()

# Histogram
plt.hist(df['Danceability'], bins=20)
plt.xlabel('Danceability')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability')
plt.show()

# Frequency of song types
df['Album_type'].value_counts().plot(kind='bar')
plt.xlabel('Album Type')
plt.ylabel('Frequency')
plt.title('Frequency of Album Types')
plt.show()

# Graph which shows the affect of energy in danceability
plt.scatter(df['Danceability'], df['Energy'])
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('Scatter Plot: Danceability vs Energy')
plt.show()


# 2D scatter plot to show 'Danceability' vs 'Energy' in a matter of color by 'Loudness'
plt.scatter(df['Danceability'], df['Energy'], c=df['Loudness'], cmap='viridis')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.title('2D Scatter Plot: Danceability vs Energy (Color by Loudness)')
plt.colorbar(label='Loudness')
plt.show()
