class EDA:

    def __init__(self, pd, np, plt, sns):
        self.pd = pd
        self.np = np
        self.plt = plt
        self.sns = sns
        self.games = pd.read_csv('dataset/games.csv')
        self.plays = pd.read_csv('dataset/plays.csv')
        self.players = pd.read_csv('dataset/players.csv')
        self.player_play = pd.read_csv('dataset/player_play.csv')

    def getDf(self, dfName):
        match dfName:
            case 'games': return self.games
            case 'plays': return self.plays
            case 'players': return self.players
            case 'player_play': return self.player_play
            case _: raise ValueError(f"Invalid data type: {dfName}")

    def describeColumns(self):
        print(f'\n{self.games.info()}\n')
        print(f'\n{self.plays.info()}\n')
        print(f'\n{self.players.info()}\n')
        print(f'\n{self.player_play.info()}\n')
        
    def removeAllNullValues(self):
        print("Droping null values cols using the dropna function")
        for name in ['games', 'plays', 'players', 'player_play']:
            print(f'\nIn {name} table\n')
            df = getDf(name)
            if df.isnull().any().any():
                print("Droping cols -> \n")
                for col in df.columns[df.isnull().any()].tolist():
                    print(col)
                df.dropna(axis=1, inplace=True)
                print("\ndropped Succesfully")
            else:
                print(f'\nNo nullable columns found  in table {name}\n')

    def descTable(self, name):
        print(f'Description of table {name}\n')
        df = getDf(name)
        # Get the number of rows and columns
        num_rows, num_cols = df.shape
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}\n")
        # Get the column names
        column_names = df.columns.tolist()
        for col in column_names:
            print(f"{col}")
        # Get a quick summary of the dataset
        summary = df.describe(include='all')
        print("\nDataset summary:")
        print(summary)

    def plottingOutliers(self, csv_name):
        data = getDf(csv_name)
        numeric_cols = data.select_dtypes(include=[self.np.number]).columns.tolist()
        # Plotting for each numeric column
        for col in numeric_cols:
            self.plt.figure(figsize=(10, 5))
            self.plt.suptitle(f"Outlier Detection for '{col}' in {csv_name}")
            # Box plot
            self.plt.subplot(1, 2, 1)
            self.sns.boxplot(data[col])
            self.plt.title(f"Box Plot of {col}")
            # Histogram
            self.plt.subplot(1, 2, 2)
            self.sns.histplot(data[col], bins=30, kde=True)
            self.plt.title(f"Histogram of {col}")
            self.plt.tight_layout()
            self.plt.show()

    def featureRelationships(self, csv_name):
        data = getDf(csv_name)
        # Detect numeric columns
        numeric_cols = data.select_dtypes(include=[self.np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            print(f"Not enough numeric columns in {csv_name} for pair plotting.")
            return
        # Pair plot
        print(f"Creating pair plot for {csv_name}...")
        self.sns.pairplot(data[numeric_cols], diag_kind='kde')
        self.plt.suptitle(f"Feature Relationships in {csv_name}", y=1.02)
        self.plt.show()

    def correlationAnalysis(self, csv_name):
        data = getDf(csv_name)
        # Select numeric columns for correlation analysis
        numeric_cols = data.select_dtypes(include=[self.np.number])
        if numeric_cols.empty:
            print(f"No numeric columns available for correlation analysis in {csv_name}.")
            return
        # Compute correlation matrix
        correlation_matrix = numeric_cols.corr()
        # Plot the heatmap
        self.plt.figure(figsize=(12, 8))
        self.sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        self.plt.title(f"Correlation Heatmap for {csv_name}", fontsize=16)
        self.plt.xticks(rotation=45)
        self.plt.yticks(rotation=0)
        self.plt.tight_layout()
        self.plt.show()

    def scatterPlots(self, csv_name, x_column, y_column):
        data = getDf(csv_name)
        # Check if columns exist in the dataframe
        if x_column not in data.columns or y_column not in data.columns:
            print(f"One or both columns '{x_column}' and '{y_column}' not found in {csv_name}.")
            return
        # Plot scatter plot
        self.plt.figure(figsize=(10, 6))
        self.sns.scatterplot(data=data, x=x_column, y=y_column)
        self.plt.title(f"Scatter Plot of {x_column} vs {y_column} in {csv_name}")
        self.plt.xlabel(x_column)
        self.plt.ylabel(y_column)
        self.plt.tight_layout()
        self.plt.show()

    def mergeDataFrames(self, left_csv, right_csv, on_column, how='inner'):
        left_data, right_data = getDf(left_csv), getDf(right_csv)
        # Check if the joining column exists in both dataframes
        if on_column not in left_data.columns or on_column not in right_data.columns:
            print(f"Joining column '{on_column}' not found in one or both datasets.")
            return
        # Perform the merge
        merged_data = self.pd.merge(left_data, right_data, on=on_column, how=how)
        print(f"DataFrames {left_csv} and {right_csv} merged on '{on_column}' using {how} join.")
        return merged_data

    def sliceData(self, csv_name, filter_column, filter_value):
        data = getDf(csv_name)
        # Check if the filter column exists in the dataframe
        if filter_column not in data.columns:
            print(f"Column '{filter_column}' not found in {csv_name}.")
            return
        # Filter the data
        filtered_data = data[data[filter_column] == filter_value]
        print(f"Data sliced from {csv_name} where {filter_column} = {filter_value}.")
        return filtered_data

