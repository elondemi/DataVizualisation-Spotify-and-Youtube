import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('Spotify_Youtube.csv')

# Define data types and assess data quality
print(df.dtypes)
print(df.describe())

# Sample 10% of the data
sampled_data = df.sample(frac=0.1)

# Handling missing values (e.g., filling missing values with the mean)
df['Danceability'].fillna(df['Danceability'].mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(subset='Track', keep='first', inplace=True)

# Select specific columns
selected_columns = df[['Track', 'Artist', 'Danceability', 'Energy']]

# Create a new column by combining existing columns
df['DanceEnergyRatio'] = df['Danceability'] / df['Energy']

# Discretize a column into intervals
bins = [0, 0.5, 1, 2]
df['DanceabilityCategory'] = pd.cut(df['Danceability'], bins)

# Apply standard scaling to a numeric column
scaler = StandardScaler()
df['ScaledEnergy'] = scaler.fit_transform(df[['Energy']])

df.to_csv('cleaned_data.csv', index=False)


# Example: Create a histogram
plt.hist(df['Danceability'], bins=20)
plt.xlabel('Danceability')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability')
plt.show()
