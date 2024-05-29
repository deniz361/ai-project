import pandas as pd 
import numpy as np
from data_preprocessing import Data_Preprocessing

Data = Data_Preprocessing(file_path='/Users/awthura/THD/ai-project/datasets/supervised_dataset.csv')
Data.normalize_data(Data.data[Data.numerical_columns])
Data = Data.preprocess_data()

Data.to_csv('/Users/awthura/THD/ai-project/datasets/supervised_dataset_processed.csv')