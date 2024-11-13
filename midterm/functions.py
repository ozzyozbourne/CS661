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

    # data description part begins 

    def descTableGame(self):
        print("Description of table game\n")

        # Get the number of rows and columns
        num_rows, num_cols = self.games.shape
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}\n")

        # Get the column names
        column_names = self.games.columns.tolist()
        for col in column_names: print(f"{col}")

        # Get a quick summary of the dataset
        summary = self.games.describe(include='all')
        print("\nDataset summary:")
        print(summary)

    def descTablePlayers(self):
        print("Description of table Players\n")

        # Get the number of rows and columns
        num_rows, num_cols = self.players.shape
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}\n")

        # Get the column names
        column_names = self.players.columns.tolist()
        for col in column_names: print(f"{col}")

        # Get a quick summary of the dataset
        summary = self.players.describe(include='all')
        print("\nDataset summary:")
        print(summary)

    def descTablePlayerPlay(self):
        print("Description of table Players Play\n")

        # Get the number of rows and columns
        num_rows, num_cols = self.player_play.shape
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}\n")

        # Get the column names
        column_names = self.player_play.columns.tolist()
        for col in column_names: print(f"{col}")

        # Get a quick summary of the dataset
        summary = self.player_play.describe(include='all')
        print("\nDataset summary:")
        print(summary)

    def descTablePlays(self):
        print("Description of table Plays\n")

        # Get the number of rows and columns
        num_rows, num_cols = self.plays.shape
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}\n")

        # Get the column names
        column_names = self.plays.columns.tolist()
        for col in column_names: print(f"{col}")

        # Get a quick summary of the dataset
        summary = self.plays.describe(include='all')
        print("\nDataset summary:")
        print(summary)

    # data description part ends    

    # data preprocessing part begins 
    
    def removeAllNullValues(self):
        print("Droping null values cols using the dropna function")

        print("\nIn games table\n")
        if self.games.isnull().any().any():
            for col in self.games.columns[self.games.isnull().any()].tolist(): print(col)
        else: print("No nullable columns found.")

        print("\nIn plays table\n")
        if self.plays.isnull().any().any():
            for col in self.plays.columns[self.plays.isnull().any()].tolist(): print(col)
        else: print("No nullable columns found.")

        print("\nIn players table\n")
        if self.players.isnull().any().any():
            for col in self.players.columns[self.players.isnull().any()].tolist(): print(col)
        else: print("No nullable columns found.")

        print("\nIn player_play table\n")
        if self.player_play.isnull().any().any():
            for col in self.player_play.columns[self.player_play.isnull().any()].tolist(): print(col)
        else: print("No nullable columns found.")


        self.games.dropna(axis=1, inplace=True)
        self.plays.dropna(axis=1, inplace=True)
        self.players.dropna(axis=1, inplace=True)
        self.player_play.dropna(axis=1, inplace=True)

        print("\ndropped Succesfully")
    # data preprocessing part ends

    #Eda begins here

    #on tables games 

    #Univariate Analysis
    def numberOfGamesPerWeekAndPlot(self):
         
        games_per_week = self.games.groupby('week')['gameId'].count()
        print("\nNumber of games per week:")
        print(games_per_week)
        
        self.plt.figure(figsize=(10, 6))
        games_per_week.plot(kind='bar', color='orange')
        self.plt.title('Number of Games Per Week')
        self.plt.xlabel('Week')
        self.plt.ylabel('Number of Games')
        self.plt.xticks(rotation=45)
        self.plt.tight_layout()  # Adjust layout to fit labels
        self.plt.show()

    def averageScoresPerWeekAndPlot(self):

        avg_scores_per_week = self.games.groupby('week')[['homeFinalScore', 'visitorFinalScore']].mean()
        print("\nAverage scores per week:")
        print(avg_scores_per_week)
        
        self.plt.figure(figsize=(10, 6))
        avg_scores_per_week.plot(kind='bar', color=['blue', 'red'], ax=self.plt.gca())
        self.plt.title('Average Scores Per Week')
        self.plt.xlabel('Week')
        self.plt.ylabel('Average Score')
        self.plt.xticks(rotation=45)
        self.plt.legend(['Home Final Score', 'Visitor Final Score'])
        self.plt.tight_layout()  # Adjust layout to fit labels
        self.plt.show()

    def plotScoreDistributionAndOutliers(self):

        # Distribution of Home Final Score
        self.plt.figure(figsize=(12, 6))
        self.sns.histplot(self.games['homeFinalScore'], kde=True, color='blue', bins=20, label='Home Final Score')
        self.plt.title('Distribution of Home Final Score')
        self.plt.xlabel('Home Final Score')
        self.plt.ylabel('Frequency')
        self.plt.legend()
        self.plt.show()

        # Distribution of Visitor Final Score
        self.plt.figure(figsize=(12, 6))
        self.sns.histplot(self.games['visitorFinalScore'], kde=True, color='red', bins=20, label='Visitor Final Score')
        self.plt.title('Distribution of Visitor Final Score')
        self.plt.xlabel('Visitor Final Score')
        self.plt.ylabel('Frequency')
        self.plt.legend()
        self.plt.show()

        # Boxplot for Home and Visitor Scores
        self.plt.figure(figsize=(12, 6))
        self.sns.boxplot(data=self.games[['homeFinalScore', 'visitorFinalScore']], orient='h', palette='Set2')
        self.plt.title('Boxplot of Home and Visitor Final Scores')
        self.plt.xlabel('Scores')
        self.plt.show()

        # Identify Outliers (IQR Method)
        for score_type in ['homeFinalScore', 'visitorFinalScore']:
            q1 = self.games[score_type].quantile(0.25)
            q3 = self.games[score_type].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = self.games[(self.games[score_type] < lower_bound) | (self.games[score_type] > upper_bound)]
            print(f"\nOutliers in {score_type}:")
            print(outliers[['gameId', score_type]])
