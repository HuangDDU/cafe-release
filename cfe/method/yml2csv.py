import pandas as pd
import yaml


def yml2csv(yaml_file, csv_file):
    # read yml file
    with open(yaml_file, "r", encoding="utf-8") as yf:
        data = yaml.safe_load(yf)

    pd.DataFrame(data).T.to_csv(csv_file)


# 使用示例
yml2csv("method_backend.yml", "method_backend.csv")
