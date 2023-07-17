import time
import datetime
import threading
import os

import pandas as pd
import numpy as np

from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup

def banking_finance():
  global dff1

  dff1 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  banking_finance = 'https://www.iimjobs.com/c/filter/banking-finance-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-13-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html'
  driver = webdriver.Chrome()
  driver.get(banking_finance)

  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(0.75)

  soup1 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup1.find('div', id='mainContainer')
  job_elems1 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem1 in job_elems1:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem1.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem1.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem1.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem1.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff1 = pd.concat([dff1, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff1.to_excel("IIMJobsJobListing_BANKING_FINANCE"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff1)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def sales_marketing():
  global dff2

  dff2 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  sales_marketing = "https://www.iimjobs.com/c/filter/sales-marketing-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-14-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"
  driver = webdriver.Chrome()
  driver.get(sales_marketing)

  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(0.75)

  soup2 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup2.find('div', id='mainContainer')
  job_elems2 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem2 in job_elems2:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem2.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem2.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem2.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem2.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff2 = pd.concat([dff2, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff2.to_excel("IIMJobsJobListing_SALES_MARKETING_"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff2)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def consulting():
  global dff3

  dff3 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  consulting = "https://www.iimjobs.com/c/filter/sales-marketing-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-14-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"
  driver = webdriver.Chrome()
  driver.get(consulting)

  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(1)

  soup3 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup3.find('div', id='mainContainer')
  job_elems3 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem3 in job_elems3:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem3.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem3.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem3.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem3.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff3 = pd.concat([dff3, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff3.to_excel("IIMJobsJobListing_consulting_"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff3)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def hr_it():
  global dff4

  dff4 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  hr_it = "https://www.iimjobs.com/c/filter/hr-ir-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-17-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"
  driver = webdriver.Chrome()
  driver.get(hr_it)

  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(1)

  soup4 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup4.find('div', id='mainContainer')
  job_elems4 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem4 in job_elems4:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem4.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem4.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem4.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem4.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff4 = pd.concat([dff4, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff4.to_excel("IIMJobsJobListing_hr_it_"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff4)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def it_systems():
  global dff5

  dff5 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  hr_it = "https://www.iimjobs.com/c/filter/it-systems-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-15-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"
  driver = webdriver.Chrome()
  driver.get(hr_it)

  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(1)

  soup5 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup5.find('div', id='mainContainer')
  job_elems5 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem5 in job_elems5:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem5.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem5.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem5.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem5.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff5 = pd.concat([dff5, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff5.to_excel("IIMJobsJobListing_it_systems_"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff5)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def scm_operations():
  global dff6

  dff6 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  scm_operations = "https://www.iimjobs.com/c/filter/scm-operations-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-19-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"
  driver = webdriver.Chrome()
  driver.get(scm_operations)
  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(1)

  soup6 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup6.find('div', id='mainContainer')
  job_elems6 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem6 in job_elems6:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem6.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem6.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem6.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem6.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff6 = pd.concat([dff6, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff6.to_excel("IIMJobsJobListing_scm_operations_"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff6)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def legal():
  global dff7

  dff7 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  legal = "https://www.iimjobs.com/c/filter/legal-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-21-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"
  driver = webdriver.Chrome()
  driver.get(legal)
  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(1)

  soup7 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup7.find('div', id='mainContainer')
  job_elems7 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem7 in job_elems7:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem7.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem7.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem7.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem7.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff7 = pd.concat([dff7, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff7.to_excel("IIMJobsJobListing_legal_"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff7)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def bpo():
  global dff8

  dff8 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  bpo = "https://www.iimjobs.com/c/filter/bpo-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-22-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"
  driver = webdriver.Chrome()
  driver.get(bpo)
  scroll = np.arange(1, 20)
  counter = 0
  
  for scroll in scroll:
    driver.execute_script("window.scrollTo(0,(document.body.scrollHeight))")
    time.sleep(1)

  soup8 = BeautifulSoup(driver.page_source,'html5lib')
  results = soup8.find('div', id='mainContainer')
  job_elems8 = results.find_all('div', class_=['col-lg-9 col-md-9 col-sm-8 container pdmobr5', 'col-lg-3 col-md-3 col-sm-4 pdlr0 mtb2 hidden-xs'])
  # print(job_elems1)

  for job_elem8 in job_elems8:
    finding = counter % 2
    if finding == 0:  
      try: 
        print(counter)
        counter = counter + 1
        
        # Title
        T_n_E = job_elem8.find('a', class_='mrmob5 hidden-xs')
        before_the_parts = T_n_E.get_text()
        parts = before_the_parts.split("(", 1)
        T = parts[0].strip()
        E = parts[1].strip(")")
        Title = T
        # print(Title)
        
        # Experience
        Exp = E
        # print(Exp)
        
        # Site
        S = 'iimjobs.com'
        Site = S
        # print(Site)

        # URL
        U = job_elem8.find('a',class_='mrmob5 hidden-xs').get('href')
        URL = U
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    else:
      try:
        print(counter)
        counter = counter + 1
        # Date Posted
        D = job_elem8.find('span', class_='gry_txt txt12 original')
        Date=D.text 
        print(Date)

        # City
        try:
          C = job_elem8.find('span')
          City=C.text
          print(City)
        except Exception as e:
          City = None
      except Exception as e:
        print("EXCEPTION OCCURRED | COUNTER = " + str(counter))
        pass
    if finding == 1:
      dff8 = pd.concat([dff8, pd.DataFrame([[Title, Exp, City, Date, Site, URL]], columns = ['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])], ignore_index=True)
      # dff8.to_excel("IIMJobsJobListing_bpo_"+ str(datetime.date.today()) + ".xlsx", index = False)
      print(dff8)
    else:
      pass
  # driver.find_element(By.XPATH, '/html/body/div[3]/div[3]/div[9]/div[5]/div/div/div[3]/div/a').click()
  time.sleep(0.5)
  driver.close()

def main():
  thread1 = threading.Thread(target=banking_finance)
  thread2 = threading.Thread(target=sales_marketing)
  thread3 = threading.Thread(target=consulting)
  thread4 = threading.Thread(target=hr_it)
  thread5 = threading.Thread(target=it_systems)
  thread6 = threading.Thread(target=scm_operations)
  thread7 = threading.Thread(target=legal)
  thread8 = threading.Thread(target=bpo)

  # Start Button Here:

  thread1.start()
  thread2.start()

  print("OPERATIONS UNDERWAY!")

  thread1.join()
  thread2.join()

  print("FIRST PHASE COMPLETED!")

  thread3.start()
  thread4.start()

  thread3.join()
  thread4.join()

  thread5.start()
  thread6.start()

  thread5.join()
  thread6.join()

  thread7.start()
  thread8.start()

  thread7.join()
  thread8.join()

  print("ALL THE DATA HAS BEEN FETCHED - NOW INTEGRATION HAS BEGUN \n *****PLEASE WAIT*****")

  dff101 = pd.DataFrame(columns=['Job Title', 'Experience Reqd', 'City', 'Date Posted', 'Site', 'URL'])

  dff101 = pd.concat([dff101, dff1], ignore_index=True)
  dff101 = pd.concat([dff101, dff2], ignore_index=True)
  dff101 = pd.concat([dff101, dff3], ignore_index=True)
  dff101 = pd.concat([dff101, dff4], ignore_index=True)
  dff101 = pd.concat([dff101, dff5], ignore_index=True)
  dff101 = pd.concat([dff101, dff6], ignore_index=True)
  dff101 = pd.concat([dff101, dff7], ignore_index=True)
  dff101 = pd.concat([dff101, dff8], ignore_index=True)

  dff101.to_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "IIMJobsJobListing_" + str(datetime.date.today()) + ".xlsx"), index=False)
  print('THE FINAL DATA IS READY')



# main()

# banking_finance = "https://www.iimjobs.com/c/filter/banking-finance-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-13-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# sales_marketing = "https://www.iimjobs.com/c/filter/sales-marketing-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-14-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# consulting = "https://www.iimjobs.com/c/filter/sales-marketing-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-14-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# hr_it = "https://www.iimjobs.com/c/filter/hr-ir-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-17-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# it_systems = "https://www.iimjobs.com/c/filter/it-systems-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-15-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# scm_operations = "https://www.iimjobs.com/c/filter/scm-operations-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-19-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# legal = "https://www.iimjobs.com/c/filter/legal-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-21-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# bpo = "https://www.iimjobs.com/c/filter/bpo-jobs-in-metros_anywhere%20in%20india_ahmedabad_amritsar_andhra%20pradesh_aurangabad_bangalore_bhubaneshwar_bihar_chandigarh_chennai_chhattisgarh_cochin-kochi_coimbatore_cuttack_dehradun_delhi_delhi%20ncr_faridabad_gandhinagar_ghaziabad_goa_greater%20noida_gujarat_guntur_gurgaon-gurugram_guwahati_haridwar_haryana_hosur_hubli_hyderabad_jaipur_jalandhar_jammu_jammu%20&%20kashmir_jamshedpur_jharkhand_jodhpur_karnataka_kerala_kolkata_lucknow_ludhiana_madurai_maharashtra_mp_mumbai_mysore_nagpur_nasik_navi%20mumbai_noida_odisha_panipat_patiala_patna_pondicherry_pune_punjab_raipur_rajasthan_rajkot_ranchi_sonipat_srinagar_surat_tamil%20nadu_telangana_thane_thiruvananthapuram_udaipur_up_uttarakhand_vadodara-baroda_varanasi-banaras_vijayawada_vishakhapatnam-vizag_warangal-22-87_88_53_45_34_79_3_65_19_14_6_64_70_84_86_58_36_1_40_55_41_13_39_8_77_37_12_57_16_71_72_4_11_46_43_42_63_20_52_31_17_5_60_48_83_9_10_2_73_66_67_68_38_18_50_47_61_85_7_15_74_33_80_62_49_44_54_32_35_69_75_51_21_59_56_81_76_78_82-0-0-1.html"

# time.sleep(3)


# driver2 = webdriver.Chrome()
# driver2.get(sales_marketing)
# soup2 = BeautifulSoup(driver2.page_source,'html5lib')

# driver3 = webdriver.Chrome()
# driver3.get(consulting)
# soup3 = BeautifulSoup(driver3.page_source,'html5lib')

# driver4 = webdriver.Chrome()
# driver4.get(hr_it)
# soup4 = BeautifulSoup(driver4.page_source,'html5lib')

# driver5 = webdriver.Chrome()
# driver5.get(it_systems)
# soup5 = BeautifulSoup(driver5.page_source,'html5lib')

# driver6 = webdriver.Chrome()
# driver6.get(scm_operations)
# soup6 = BeautifulSoup(driver6.page_source,'html5lib')

# driver7 = webdriver.Chrome()
# driver7.get(legal)
# soup7 = BeautifulSoup(driver7.page_source,'html5lib')

# driver8 = webdriver.Chrome()
# driver8.get(bpo)
# soup8 = BeautifulSoup(driver8.page_source,'html5lib')

# print (soup1.prettify(), soup2.prettify())