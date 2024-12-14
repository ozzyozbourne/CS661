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
        self.diabetes = self.loadCsvFile('diabetes.csv')


    def printShapeColumnsTypes(self):
        print(f"Shape is -> \n{self.diabetes.shape}\n")
        print(f"Columns are -> \n{self.diabetes.columns}\n")
        print(f"Types are -> \n{self.diabetes.dtypes}\n")

    def printDescriptionNullableCols(self):
        print(f"\n{self.diabetes.describe().T}\n")
        print(f"\n{self.diabetes.isnull().any()}\n")

