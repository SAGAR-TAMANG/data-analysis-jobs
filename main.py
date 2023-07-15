import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from textblob import Word

nltk.download('punkt')
nltk.download('stopwords')

naukri=pd.read_excel('NaukriJobListing_2023-07-14.xlsx')
job_titles=pd.read_excel('job-titles.xlsx')


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Load the stopwords set
    stop_words = set(stopwords.words('english'))
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    # Rejoin tokens into a string
    processed_text = ' '.join(tokens)
    return processed_text



def working_null_values():
  print(naukri.isnull().sum())
  # naukri.fillna('NULL')

def job_title_analysis_naukri(df, title_df):
  df['Job Title'] = naukri['Job Title'].apply(preprocess_text)

  # Define the text to compare
  df['Job Title'] = df['Job Title'].str.lower()

  for idx, job_desc in df.iterrows():
    processed_desc = job_desc['processed_description']
    similarity_scores = title_df['processed_title'].apply(lambda x: cosine_similarity([processed_desc], [x])[0][0])
    max_similarity = max(similarity_scores)
    most_similar_title = title_df.loc[similarity_scores.idxmax(), 'Title']
    df.at[idx, 'Job Title'] = most_similar_title

    print(df)

  print("\n BEFORE \n")
  df_name.drop(['Rating'], axis=1, inplace=True)

def exporting_excel(df_name):
  df_name.to_excel(df_name + '.xlsx', index = False)

def similarity(df, titles):
  # Load the DataFrame with the 'Job Title' column
  # Load the file with the bunch of words
  # with open('bunch_of_words.txt', 'r') as file:
  #     bunch_of_words = [line.strip() for line in file]

  # Compute similarity scores for each word in the 'Job Title' column
  df['Similarity'] = df['Job Title'].apply(lambda x: max(Word(word).wup_similarity(Word(x)) for word in titles))

  # Replace 'Job Title' with the most similar words from the bunch
  df['Job Title'] = df['Job Title'].apply(lambda x: max(titles, key=lambda word: Word(word).wup_similarity(Word(x))))

  # Display the updated DataFrame
  print(df)


# print(job_title_analysis_naukri(naukri))

# job_title_analysis_naukri(naukri, job_titles)
# print(preprocess_text(naukri['Job Title']))

# sns.countplot(x = 'Date Posted', data = naukri2)
# plt.show()
# sns.countplot(x = 'Job Title', data = naukri2)
# plt.show()

similarity(naukri, job_titles)