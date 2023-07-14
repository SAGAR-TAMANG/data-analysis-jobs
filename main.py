import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

naukri=pd.read_excel('NaukriJobListing_2023-07-13.xlsx')

counter = 1
# unique_title = len(naukri['Job Title'].unique())
# print("Number of unique titles: " + str(unique_title))

# duplicate_title = naukri['Job Title'].duplicated()
# print(naukri.describe)

print(naukri.isnull().sum())
# naukri.fillna('NULL')

naukri2 = naukri.copy()

print("\n BEFORE \n")
naukri2.drop(['Rating'], axis=1, inplace=True)

# print(naukri2.isnull().sum())
print(naukri2.describe())

naukri2.describe().to_excel('naukri2.xlsx', index = False)

# sns.countplot(x = 'Date Posted', data = naukri2)
# plt.show()
# sns.countplot(x = 'Job Title', data = naukri2)
# plt.show()


# Number of jobs today

# Number of jobs for graduates today

# Number of jobs for post-graduates today 
