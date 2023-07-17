import re
import string
import datetime
import os
import sys
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

cur_path = os.path.dirname(os.path.abspath(__file__))

scripts = os.path.join(cur_path, 'scripts')
data = os.path.join(scripts, 'data')

# sys.path.append(scripts)
## Turn these on later:
# nltk.download('punkt')
# nltk.download('stopwords')

try:
  naukri=pd.read_excel(os.path.join(data, "NaukriJobListing_" + str(datetime.date.today()) + ".xlsx"))
except Exception as e:
  print('EXCEPTION OCCURED \n DATA NOT PRESENT \n \n FETCHING DATA NOW, PLEASE WAIT...')
  naukri = subprocess.run(['python', os.path.join(scripts, 'naukri scraper.py')])
  naukri.wait()
  naukri=pd.read_excel(os.path.join(data, "NaukriJobListing_" + str(datetime.date.today()) + ".xlsx"))

try:
  iimjobs=pd.read_excel(os.path.join(scripts, "IIMJobsJobListing_" + str(datetime.date.today()) + ".xlsx"))
except Exception as e:
  print('EXCEPTION OCCURED \n DATA NOT PRESENT \n \n FETCHING DATA NOW, PLEASE WAIT...')
  iimjobs = subprocess.run(['python', os.path.join(scripts, 'iimjobs scraper.py')])
  iimjobs.wait()
  iimjobs=pd.read_excel(os.path.join(scripts, "IIMJobsJobListing_" + str(datetime.date.today()) + ".xlsx"))

job_titles=pd.read_excel(os.path.join(data, "job-titles.xlsx"))

def frequency_title(df, column):
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
  
  # df = pd.read_excel('NaukriJobListing_2023-07-14.xlsx')
  dt = df[column].apply(cleaning)

  from collections import Counter
  p = Counter(" ".join(dt).split()).most_common(4)
  rslt = pd.DataFrame(p, columns=['Word', 'Frequency'])
  return(rslt)

def exporting_excel(df_name):
  df_name.to_excel(df_name + '.xlsx', index = False)

def count_min_exp(df):
  global zero, three, five 
  global zero_sal, three_sal, five_sal
  zero_sal = 0
  three_sal = 0
  five_sal = 0
  exp_initial_fetcher = lambda text: text.split('-')[0]
  for index, row in df.iterrows():
    var = row['Experience Reqd']
    
    if (var != 'Not-Mentioned'):
      initial_val = exp_initial_fetcher(var)
      initial_val = int(initial_val)
      if initial_val == 0:
        zero_sal += 1
      elif initial_val >= 3 and initial_val < 5:
        three_sal += 1
      elif initial_val >= 5:
        five_sal += 1
  print('ZERO :', zero_sal)
  print('THREE :', three_sal)
  print('FIVE :', five_sal)

def date_posted(df, type):
  global today, week, others
  today = 0
  week = 0
  others = 0
  if type == 'naukri':
    print('NAUKRI INITIATED')
    for index, row in df.iterrows():
      dates = row['Date Posted'].strip()
      try:
        if dates == 'Just Now':
          today += 1
          week += 1
        elif dates == '1 Day Ago' or dates == '2 Days Ago'  or dates == '3 Days Ago' or dates == '4 Days Ago' or dates == '5 Days Ago' or dates == '6 Days Ago':
          week += 1
        else:
          others += 1
      except Exception as e: 
        print('EXCEPTION OCCURED DURING THE RUNNING OF def date_posted')
  elif type == 'iimjobs':
    print('IIM JOBS INITIATED')
    current_date = datetime.date.today()
    y0 = current_date.strftime("%d/%m/%Y")
    for index, row in df.iterrows():
      date = row['Date Posted'].strip()
      y1 = current_date - datetime.timedelta(days=1)
      y1 = y1.strftime("%d/%m/%Y")
      y2 = current_date - datetime.timedelta(days=2)
      y2 = y2.strftime("%d/%m/%Y")
      y3 = current_date - datetime.timedelta(days=3)
      y3 = y3.strftime("%d/%m/%Y")
      y4 = current_date - datetime.timedelta(days=4)
      y4 = y4.strftime("%d/%m/%Y")
      y5 = current_date - datetime.timedelta(days=6)
      y5 = y5.strftime("%d/%m/%Y")
      if date == y0:
        today+=1
        week+=1
      elif date == y1 or date == y2 or date == y3 or date == y4 or date == y5:
        week+=1
      else:
        others+=1
      
## RUNNING THE FUNCTIONS:

f = frequency_title(naukri, 'Job Title')

# count_min_exp(naukri)
# date_posted(iimjobs, 'iimjobs')
# print("TODAY :", today)
# print("THIS WEEK :", week)
# print('OTHERS :', others)

# date_posted(naukri, 'naukri')
# print("TODAY :", today)
# print("THIS WEEK :", week)
# print('OTHERS :', others)
## DISPLAYING THE FUNCTIONS

# sns.barplot(x = 'Word',y = 'Frequency', data = f)
# plt.show()
print(f)

## Doing the coding stuff:

# def main():
  
