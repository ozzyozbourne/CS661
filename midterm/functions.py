import sys


class EDA:
    # Error handling
    def loadCsvFile(self, filepath):
        try:
            df = self.pd.read_csv(filepath)
            print(f"Successfully loaded {filepath}")
            return df
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found. Please check the path.")
            sys.exit(1)

    def __init__(self, pd, np, plt, sns):
        self.pd = pd
        self.np = np
        self.plt = plt
        self.sns = sns
        self.games = self.loadCsvFile('dataset/games.csv')
        self.plays = self.loadCsvFile('dataset/plays.csv')
        self.players = self.loadCsvFile('dataset/players.csv')
        self.player_play = self.loadCsvFile('dataset/player_play.csv')

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
            df = self.getDf(name)
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
        df = self.getDf(name)
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

    # Step 5
    def plottingOutliers(self, csv_name):
        data = self.getDf(csv_name)
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

    # Step 6
    def featureRelationships(self, csv_name):
        data = self.getDf(csv_name)
        # Detect numeric columns
        numeric_cols = data.select_dtypes(include=[self.np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            print(f"Not enough numeric columns in {csv_name} for pair plotting.")
            return
        numeric_cols = numeric_cols[:10]  # Limit to the first 20 numeric columns
        # Pair plot
        print(f"Creating pair plot for {csv_name}...")
        self.sns.pairplot(data[numeric_cols], diag_kind='kde')
        self.plt.suptitle(f"Feature Relationships in {csv_name}", y=1.02)
        self.plt.show()

    # Step 7
    def correlationAnalysis(self, csv_name):
        data = self.getDf(csv_name)
        # Select numeric columns for correlation analysis
        numeric_cols = data.select_dtypes(include=[self.np.number])
        if numeric_cols.empty:
            print(f"No numeric columns available for correlation analysis in {csv_name}.")
            return
        numeric_cols = numeric_cols.iloc[:, :10]  # Limit to the first 20 numeric columns
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

    # Step 8
    def scatterPlots(self, csv_name, x_column, y_column):
        data = self.getDf(csv_name)
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

    # Step 9
    def mergeDataFrames(self, left_csv, right_csv, on_column, how='inner'):
        left_data, right_data = self.getDf(left_csv), self.getDf(right_csv)
        # Check if the joining column exists in both dataframes
        if on_column not in left_data.columns or on_column not in right_data.columns:
            print(f"Joining column '{on_column}' not found in one or both datasets.")
            return
        # Perform the merge
        merged_data = self.pd.merge(left_data, right_data, on=on_column, how=how)
        print(f"DataFrames {left_csv} and {right_csv} merged on '{on_column}' using {how} join.")
        return merged_data

    # Step 10
    def sliceData(self, csv_name, filter_column, filter_value):
        data = self.getDf(csv_name)
        # Check if the filter column exists in the dataframe
        if filter_column not in data.columns:
            print(f"Column '{filter_column}' not found in {csv_name}.")
            return
        # Filter the data
        filtered_data = data[data[filter_column] == filter_value]
        print(f"Data sliced from {csv_name} where {filter_column} = {filter_value}.")
        return filtered_data

    # 11. Matrix Representation
    def matrixRepresentation(self, data):
        data = self.getDf(data)
        if isinstance(data, self.pd.DataFrame):
            matrix = data.values
            print("Matrix Representation:")
            print(matrix)
            return matrix
        else:
            print("Input must be a pandas DataFrame.")
            return None

    # 12. NumPy Integration
    def toNumpyArray(self, data):
        data = self.getDf(data)
        if isinstance(data, self.pd.DataFrame):
            numpy_array = data.to_numpy()
            print("Converted to NumPy Array:")
            print(numpy_array)
            return numpy_array
        else:
            print("Input must be a pandas DataFrame.")
            return None

    # 13. Data Selection
    def selectData(self, data, rows=None, cols=None):
        data = self.getDf(data)
        if isinstance(data, self.pd.DataFrame):
            # Select specific rows and columns
            selected_data = data.iloc[rows, cols] if rows is not None or cols is not None else data
            print("Selected Data Slice:")
            print(selected_data)
            return selected_data
        else:
            print("Input must be a pandas DataFrame.")
            return None

    # 16
    def filterData(self, data, conditions):
        data = self.getDf(data)
        if not isinstance(data, self.pd.DataFrame):
            print("Input must be a pandas DataFrame.")
            return None

        filtered_data = data.copy()
        for column, (operator, value) in conditions.items():
            if column not in filtered_data.columns:
                print(f"Column '{column}' not found in the DataFrame.")
                continue

            # Apply filtering based on the operator
            if operator == '>':
                filtered_data = filtered_data[filtered_data[column] > value]
            elif operator == '<':
                filtered_data = filtered_data[filtered_data[column] < value]
            elif operator == '>=':
                filtered_data = filtered_data[filtered_data[column] >= value]
            elif operator == '<=':
                filtered_data = filtered_data[filtered_data[column] <= value]
            elif operator == '==':
                filtered_data = filtered_data[filtered_data[column] == value]
            elif operator == '!=':
                filtered_data = filtered_data[filtered_data[column] != value]
            else:
                print(f"Unsupported operator '{operator}' for column '{column}'. Skipping...")

        print("Filtered Data:")
        print(filtered_data)

    #Eda begin here
    def mergeTheDataSets(self):
        # Merge player_play with plays to get play details
        self.player_play_merged = self.player_play.merge(self.plays, on=['gameId', 'playId'])
        # Merge with players to get player details
        self.player_play_merged = self.player_play_merged.merge(self.players, on='nflId')
        # Merge with games to get game outcomes
        self.combined_data = self.player_play_merged.merge(self.games, on='gameId')

    def keyMetricsAndFeatures(self):
        # Calculate total rushing yards and passing yards per player
        aggregate_metrics = self.combined_data.groupby('nflId').agg(
            total_rushing_yards=self.pd.NamedAgg(column='rushingYards', aggfunc='sum'),
            total_passing_yards=self.pd.NamedAgg(column='passingYards', aggfunc='sum'),
            total_plays=self.pd.NamedAgg(column='playId', aggfunc='count')
        ).reset_index()

        # Count the number of plays for each game without conflicting names
        play_counts = self.combined_data.groupby('gameId')['playId'].count().reset_index(name='play_count')

        # Merge play counts back into the combined_data (avoid naming conflicts)
        self.combined_data = self.combined_data.merge(play_counts, on='gameId', how='left', suffixes=('', '_y'))

        # Calculate average yards gained per play, handle division by zero
        self.combined_data['average_yards_gained'] = self.combined_data['yardsGained'] / self.combined_data['play_count']

        # Replace infinity values with 0
        self.combined_data['average_yards_gained'].replace([float('inf'), -float('inf')], 0, inplace=True)

        # Create additional features: total scores per game
        total_scores = self.combined_data.groupby('gameId').agg(
            home_final_score=self.pd.NamedAgg(column='homeFinalScore', aggfunc='first'),
            visitor_final_score=self.pd.NamedAgg(column='visitorFinalScore', aggfunc='first')
        ).reset_index()

        # Rename the score columns to avoid conflicts
        total_scores.rename(columns={
            'home_final_score': 'home_final_score_new',
            'visitor_final_score': 'visitor_final_score_new'
        }, inplace=True)

        # Merge total scores back into the combined data
        self.combined_data = combined_data.merge(total_scores, on='gameId', how='left')

        # Display the final aggregated metrics and the combined dataset
        print(aggregate_metrics.head())

    def printDetails(self):
        print(self.combined_data[['gameId', 'nflId', 'average_yards_gained', 'home_final_score_new', 'visitor_final_score_new']].head())
        print(data.info())
        print(data.describe())

