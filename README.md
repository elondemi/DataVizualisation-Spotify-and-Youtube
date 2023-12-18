# Data Visualization: Spotify & YouTube

<!-- Brief Project Description and its Purpose. -->

## Initial Data Preprocessing Requirements

- Data Collection
- Data Type Definition
- Data Quality
- Integration, Aggregation, and Display
- Data Cleaning
- Dimension Reduction
- Subset Selection
- Feature Engineering
- Discretization and Binarization
- Handling Missing Values Strategy

The first part of this project aims to preprocess the data to enable a more robust and accurate analysis. To start with preprocessing, you can follow the steps and tasks mentioned above to prepare the data for your further analysis.

## Usage

### Requirements:
- Ensure you have the following libraries installed: `pandas`, `scikit-learn`, `matplotlib`.
- The dataset file `Spotify_Youtube.csv` should be located in the directory: `Data-Visualization-Spotify-and-Youtube/src`.

### Steps:

1. **Import Required Libraries:**
   ```
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   from sklearn.decomposition import PCA
   import matplotlib.pyplot as plt
   ```

2. **Load Dataset:**
   ```
   df = pd.read_csv('Spotify_Youtube.csv')
   ```

3. **Understanding Data:**
   - Display data types and summary statistics of the dataset:
     ```
     print(df.dtypes)
     print(df.describe())
     ```

4. **Data Sampling and Cleansing:**
   - Sample 10% of the data for analysis:
     ```
     sampled_data = df.sample(frac=0.1)
     ```
   - Handle missing values in the 'Danceability' column by filling them with the mean:
     ```
     df['Danceability'].fillna(df['Danceability'].mean(), inplace=True)
     ```
   - Remove duplicate rows based on the 'Track' column, keeping the first occurrence:
     ```
     df.drop_duplicates(subset='Track', keep='first', inplace=True)
     ```
   - Select specific columns for further analysis:
     ```
     selected_columns = df[['Track', 'Artist', 'Danceability', 'Energy']]
     ```
   - Perform feature engineering to create new columns and dicretization:
     ```
     df['DanceEnergyRatio'] = df['Danceability'] / df['Energy']
     df['DanceabilityCategory'] = pd.cut(df['Danceability'], bins=[0, 0.5, 1, 2])
     scaler = StandardScaler()
     df['ScaledEnergy'] = scaler.fit_transform(df[['Energy']])
     df['SongLength'] = df['Duration_ms'] / 1000  # Convert milliseconds to seconds
     df['HighEnergy'] = df['Energy'].apply(lambda x: 1 if x > 0.8 else 0)
     ```

5. **Data Visualization:**
   - Plot a histogram showing the distribution of 'Danceability':
     ```
     plt.hist(df['Danceability'], bins=20)
     plt.xlabel('Danceability')
     plt.ylabel('Frequency')
     plt.title('Distribution of Danceability')
     plt.show()
     ```

6. **Dimensionality Reduction using PCA:**
   - Prepare features for dimensionality reduction:
     ```
     features = df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Tempo']]
     features = features.fillna(features.mean())
     ```
   - Perform PCA to reduce features to 2 principal components:
     ```
     pca = PCA(n_components=2)
     reduced_features = pca.fit_transform(features)
     reduced_df = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])
     ```

    Reducted 5 fields: 'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Tempo' into PC1 and PC2

```	
               PC1	       PC2
    0	      17.938889	     -0.580238
    1	      -27.826627    -2.505957
    2	      -12.533721    -4.042393
    3	      -0.171957	    -1.869341
    4	      47.279736	    2.051632
    ...	      ...	    ...
    20713	-30.596875	-2.047945
    20714	54.353077	  -4.628654
    20715	47.806536	  -1.884195
    20716	34.815720	  -2.858737
    20717	39.491057	  -2.217929
```

### Execution in Jupyter Notebook:

- Place the `preprocessingdata.py` file inside the `src` folder of your Jupyter environment.
- Ensure that the `Spotify_Youtube.csv` dataset file is located in the directory specified (`Data-Visualization-Spotify-and-Youtube/src`).

Execute the code within the Jupyter Notebook by running each section sequentially. This process will:
- Load the dataset.
- Perform data cleaning, sampling, and feature engineering.
- Visualize the distribution of 'Danceability'.
- Apply PCA for dimensionality reduction.
- Save the cleaned data into a CSV file named `cleaned_data.csv`.

Adjust paths and filenames if needed based on your specific directory structure and file locations.

## Contributors

- [Lorent Sinani](https://github.com/lorentsinani)

- [Elon Demi](https://github.com/elondemi)
