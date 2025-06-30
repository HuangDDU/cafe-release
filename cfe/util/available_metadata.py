import os

import pandas as pd
import yaml


def get_available_method_df():
    # read available methods from config yml file
    with open(os.path.join(os.path.dirname(__file__), "../method/method_backend.yml"), 'r') as file:
        method_backend_dict = yaml.safe_load(file)

    df = pd.DataFrame(method_backend_dict).T
    return df


def get_available_dataset_df():
    # TODO: read available methods from py file
    df = pd.DataFrame()
    return df
