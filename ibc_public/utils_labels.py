import json
import sys
import pandas as pd

import os


class LabelsDb:
    """
    Opens a .tsv files containing contrasts and cognitive labels, and allows
    to retrieve information from it, as well as adding new labels to existing
    contrasts

    Attributes
    ----------

    path: str, Default None
          Path to file

    contrast_col: str, default 'contrast'
                  Name of the column of your dataframe that contains contrast
                  information

    sep: str, default '\t'
         Separator used in your file, to be passed to pandas

    data: pd.DataFrame
          DataFrame object containing all the information
    """

    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    def __init__(self, path: str = None, contrast_col: str = 'contrast',
                 sep: str = '\t'):
        self.path = path
        if not self.path:
            self.path = self._load_config(self.config_path)
        self.contrast_col = contrast_col
        self.sep = sep
        self.data = pd.read_csv(self.path, sep=self.sep)

    def _load_config(self, path: str) -> str:
        """Looks for a config file in the working directory"""
        while True:
            try:
                with open(path, 'r') as config_file:
                    config = json.load(config_file)
                    return config['default_path']
            except FileNotFoundError as err:
                print(f"\nNo path was provided and no config file was found.")
                choice = input(f"Do you want to create one? [y/n]: ")
                if choice == 'y':
                    self._create_config_file()
                elif choice == 'n':
                    print(f"\nNo path or configuration file "
                          f"were found. Exiting...")
                    sys.exit()
                else:
                    print("Please answer 'y' or 'n' \n")

    def _create_config_file(self):
        """Create a configuration file with the path to your db"""
        print(f"\nA config file will be created at {self.config_path}")
        while True:
            user_path = input("Please, provide the path for your file: ")
            if os.path.exists(user_path):
                config = {"default_path": user_path}
                with open(self.config_path, 'w') as config_file:
                    json.dump(config, config_file)
                    break
            else:
                print(f"No such file or directory: {user_path}. Please "
                      f"try again")

        print(f"\nConfig file created at {self.config_path}. This will be the"
              f"file loaded every time you instantiate this object without "
              f"passing any path.")

    def get_labels(self, *contrasts: str):
        """Returns the list of labels for each passed contrast name"""
        for contrast in contrasts:
            df = self.data
            con = df[df[self.contrast_col] == contrast]

            if len(con.index) == 0:
                print(f"There is no contrast with the name {contrast}")
            else:
                labels = df.columns[con.isin([1.0]).any()]
                print(f"The labels for {contrast} are: "
                      f"{[label for label in labels]}")

    def add_labels(self, contrast: str, *labels: str):
        """Adds all the passed labels to the selected contrast"""
        df = self.data
        con_index = df[df[self.contrast_col] == contrast].index
        for label in labels:
            if label in df.columns:
                df.at[con_index, label] = 1.0
            else:
                print(f"There is no label with the name {label}")

    def save_db(self, path: str = None):
        if not path:
            path = self.path
        self.data.to_csv(path, sep=self.sep, index=False)
