import pandas as pd
from sklearn.preprocessing import StandardScaler
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

df.to_csv('cleaned_data.csv', index=False)


# Histogram
plt.hist(df['Danceability'], bins=20)
plt.xlabel('Danceability')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability')
plt.show()
