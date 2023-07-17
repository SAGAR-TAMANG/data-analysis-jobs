import requests
import time
import datetime
import os

import pandas as pd
import numpy as np

from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup

def main():
  dff = pd.DataFrame(columns=['Job Title','Description', 'Experience Reqd', 'Company', 'City', 'Salary Range', 'Date Posted', 'Rating', 'Site', 'URL'])

  url = "https://www.naukri.com/jobs-in-india"
  # Observation: Page1: https://www.naukri.com/it-jobs?k=it Page2: https://www.naukri.com/it-jobs-2

  page = requests.get(url)
  # print(page.text)

  # Making the Driver Headless: (next 3 lines)
  driver = webdriver.Chrome()

  # time.sleep(3)

  # driver = webdriver.Chrome()
  driver.get(url)
  time.sleep(3)
  driver.find_element(By.XPATH, '//*[@id="root"]/div[4]/div[1]/div/section[2]/div[1]/div[2]/span/span[2]/p').click()
  driver.find_element(By.XPATH, '//*[@id="root"]/div[4]/div[1]/div/section[2]/div[1]/div[2]/span/span[2]/ul/li[2]').click()

  time.sleep(5)
  pages = np.arange(1,50)

  for pages in pages:
    soup = BeautifulSoup(driver.page_source,'html5lib')
    results = soup.find(class_='list')
    job_elems = results.find_all('article', class_='jobTuple')
    for job_elem in job_elems:
      # Post Title
      T = job_elem.find('a',class_='title ellipsis')
      Title=T.text
      # print("Job Title: " + Title.text)
      # print(Title)

      try:
        i_tag = job_elem.find('i', 'fleft naukicon naukicon-srp-description')
        D = i_tag.next_sibling.strip()
        Description = D
      except Exception as e:
        Description = None
      # Experience  
      E = job_elem.find('span', class_='ellipsis fleft expwdth')
      if E is None:
        Exp = "Not-Mentioned"
      else:
        Exp = E.text
      print(Exp)
      # print('Experience: ' + Exp.text)
      # print(" "*2)

      # Company
      C = job_elem.find('a', class_='subTitle ellipsis fleft')
      Company=C.text
      # print("Company: " + Company.text)
      
      # City
      try:
        C = job_elem.find('span', class_='ellipsis fleft locWdth')
        City=C.text
      except Exception as e:
        City = None

      # Salary Range
      S = job_elem.find('span', class_='ellipsis fleft')
      Salary=S.text

      # Date Posted
      D = job_elem.find('span', class_='fleft postedDate')
      if D == 'Just Now':
        Date = 'Today'
      else:
        Date=D.text

      # Rating
      try:
        i_tag = job_elem.find('i', 'naukicon naukicon-rating-yellow-star fright')
        R1 = i_tag.next_sibling.strip()
        Rating = R1
      except Exception as e:
        Rating = None

      # Site
      S = 'Naukri.com'
      Site=S

      U = job_elem.find('a',class_='title ellipsis').get('href')
      URL = U
      # print(URL)

      # df=df.append({'Title':Title, 'Company':Company,'URL':URL}, ignore_index = True)
      # df = pd.DataFrame([[Title, Company, URL]], columns=['Title','Company','URL'])
      dff = pd.concat([dff, pd.DataFrame([[Title, Description, Exp, Company, City, Salary, Date, Rating, Site, URL]], columns = ['Job Title','Description', 'Experience Reqd', 'Company', 'City', 'Salary Range', 'Date Posted', 'Rating', 'Site', 'URL'])], ignore_index=True)
      print(dff)
      # Second Way using Concat:
      # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
      # df3 = pd.concat([df3, df2], ignore_index=True)
      # dff.to_csv("Naukri.com_Data_Collection.csv", index = False)
      dff.to_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "NaukriJobListing_" + str(datetime.date.today()) + ".xlsx"), index=False)

    
      
    # time.sleep(0.5)

    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight) - 1500)")

    # time.sleep(0.75)

    # script = 'your JavaScript goes here'
    # element = driver.find_element_by_*('your element identifier goes here')
    # driver.execute_script(script, element)
    driver.find_element(By.XPATH, '/html/body/div[1]/div[4]/div/div/section[2]/div[3]/div/a[2]  ').click()
    # /html/body/div[1]/div[4]/div/div/section[2]/div[3]/div/a[2]
    # //*[@id="root"]/div[4]/div/div/section[2]/div[3]/div/a[2]

    time.sleep(3) # Increase the sleep time if facing the slowness in loading of the next page

  print("*********************CONCLUSION: FINISHED FETCHING DATA FROM NAUKRI.COM*********************")

  # Closing the Driver
  driver.close()

main()