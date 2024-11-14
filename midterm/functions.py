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
            case _: raise ValueError(f"Invalid data type: {data_type}")

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
                print("No nullable columns found.")

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

