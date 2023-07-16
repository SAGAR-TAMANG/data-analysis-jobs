import re
import string
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from textblob import Word
from collections import Counter
from nltk.corpus import stopwords


## Turn these on later:
# nltk.download('punkt')
# nltk.download('stopwords')

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

def frequency(df):
  stop_words = stopwords.words()

  def cleaning(text):        
      # converting to lowercase, removing URL links, special characters, punctuations...
      text = text.lower()
      text = re.sub('https?://\S+|www\.\S+', '', text)
      text = re.sub('<.*?>+', '', text)
      text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
      text = re.sub('\n', '', text)
      text = re.sub('[’“”…]', '', text)     

      # removing the emojies               # https://www.kaggle.com/alankritamishra/covid-19-tweet-sentiment-analysis#Sentiment-analysis
      emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
      text = emoji_pattern.sub(r'', text)   
      
      # removing the stop-words          
      text_tokens = word_tokenize(text)
      tokens_without_sw = [word for word in text_tokens if not word in stop_words]
      filtered_sentence = (" ").join(tokens_without_sw)
      text = filtered_sentence
      
      return text
  df = pd.read_excel('NaukriJobListing_2023-07-14.xlsx')
  dt = df['Job Title'].apply(cleaning)

  from collections import Counter
  p = Counter(" ".join(dt).split()).most_common(5)
  rslt = pd.DataFrame(p, columns=['Word', 'Frequency'])
  return(rslt)

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

f = frequency(naukri)
print(f)

sns.barplot(x = 'Word',y = 'Frequency', data = f)
plt.show()
