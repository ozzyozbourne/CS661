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
