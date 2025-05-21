import numpy as np
import pandas as pd

dataset_original_titanic = pd.read_excel('./media/titanicOriginal.xlsx')

dataset_cleaned_titanic = pd.read_excel('./media/titanicdatacleaned.xlsx')

dataset_original_titanic= dataset_original_titanic.head(10)

dataset_cleaned_titanic= dataset_cleaned_titanic.head(10)