def avg_sal(df, column_name):
  df.apply(count_min_exp)
  for row in df.iterrows():
    var = row[column_name]


avg_sal(naukri, 'Salary Range')


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