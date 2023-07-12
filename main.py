import pandas as pd
import numpy as np

naukri=pd.read_excel('NaukriJobListing_2023-07-11.xlsx')

counter = 1
unique_title = naukri['Job Title']

print("unique_title")


# Number of jobs today

# Number of jobs for graduates today

# Number of jobs for post-graduates today 

chunk_size = 10000

# Initialize an empty set to store unique job titles
unique_job_titles = set()

# Iterate through the dataset in chunks
for chunk in pd.read_excel('NaukriJobListing_2023-07-11.xlsx', chunksize=chunk_size):
    # Extract unique job titles from the current chunk
    unique_job_titles.update(chunk['Job Title'].unique())

# Calculate the count of unique job titles
num_unique_job_titles = len(unique_job_titles)

print("The number of unique job titles is:", num_unique_job_titles)
