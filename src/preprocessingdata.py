import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


# Histogram
plt.hist(df['Danceability'], bins=20)
plt.xlabel('Danceability')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability')
plt.show()
