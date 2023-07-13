import pandas as pd
import numpy as np

naukri=pd.read_excel('NaukriJobListing_2023-07-11.xlsx')

counter = 1
# unique_title = len(naukri['Job Title'].unique())
# print("Number of unique titles: " + str(unique_title))

duplicate_title = naukri['Job Title'].duplicated().sum()
print(duplicate_title)

# print(naukri.isnull().any().sum())
# naukri.fillna(value=None, method=None)
# print(naukri.isnull().any().sum())

# Number of jobs today

# Number of jobs for graduates today

# Number of jobs for post-graduates today 
