class EDA:

    def __init__(self, pd, np, plt, sns, games, plays, players, player_play):
        self.pd = pd
        self.np = np
        self.plt = plt
        self.sns = sns
        self.games = pd.read_csv(games)
        self.plays = pd.read_csv(plays)
        self.players = pd.read_csv(players)
        self.player_play = pd.read_csv(player_play)


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
