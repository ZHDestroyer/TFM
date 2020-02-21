# -*- coding: utf-8 -*-



"""
Master_feature_engineer:
    -Function file to quickly use keywords selection to define the final master 
    data set that is being used inside the Neural Net / deep learning 

INPUT:  I_data_path       stg. file path and name of final merged master data  
        I_keyarray        Array list of 10 numbers for the keylist hihglighted 
                          below to choose segmentation of data          
        I_out_filename    stg. filename only of output csv for UV master data 

3, 5, 6, 5, 4, 2, 4, 2,
key_age = 1
# 1: C0 = All ages
# 2: New columns: 0 to 14, 15 to 19, 20 to 29, 30 to 44, 45 to 59, 60 to 74 and 75 and over
# 3: C1,...,C16 = Age0_4,...,Age90+


key_CountryBirth = 1
# 1: C0 = All
# 2: C1,C7, C8, C11 = UK, Ireland, EU, RoW
# 3: C1, C7+C8 (=C12), C11 = UK, Ral EU, RoW
# 4: C1,C7,C9,C10,C11 = UK, Ireland, EU2001-, EU2001+, RoW
# 5: C2,...,C7 and C9,...,C11 = England,..., Wales and EU2001-, EU2001+, RoW

key_economicActivity = 1
# 1: C0 = Economic Activity's All persons
# 2: C18 and C36 = Economic Activity by Male and Female
# 3: C2, C6+C7 and C8 = Total empl. + not empl + inact. total
# 4: C20, C24+C25, C26, C42+C43 and C44 = Total empl. + not empl + inact. total by Male and Female
# 5: C3,...,C7 and C9,...,C13 = All subcat minus total average column
# 6: C21,...,C25, C27,...,C31, C39,...,43 and C45,...,C49 = All subcat minus total average column by Male and Female

key_health = 1
# 1: C0 = All
# 2: C1,...,C3 = limited a lot, little and not limited
# 3: C7,...,C11 = very good, good, fair, bad and very bad health
# 4: C12,...,C15 = Provides no unpaid care, Provides 1 to 19 hours/week,...,Provides 50 or more hours a week
# 5: C4,..,C6 and C4_star,...,C6_star  = limited a lot, little and not limited by Age  (Age 16 to 64 and Age 0 to 15/Age 65+)
#      with (e.g) C4_star = C1 - C4

key_hoursWorked = 1
# 1: C0 = All
# 2: C1,...,C4 = Part-time: 15 hours,...,49 or more hours
# 3: C5 and C10 = All by Male and Female
# 4: C6,...,C9 and C11,...,C14 = Part-time: 15 hours,...,49 or more hours by Male and Female

key_Language = 1
# 1: C0 = All
# 2: C1,...,C4 = All people aged 16 and over in household have English as a main language ,...,No people in household have English as a main language

key_Occupation = 1
# 1: C0 = All 
# 2: C1,...,C9 = Managers, directors and senior officials,..., Elementary occupations
# 3: C10 and C20 = All by Male and Female
# 4: C11,...,C19 and C21,...,C29 = Managers, directors and senior officials,..., Elementary occupations by Male and Female

key_Passport = 1
# 1: C0 and C12 = All, People with dual nationality
# 2: C1,...,C11 and C12 =  No passport,..., passport from Antarctica and Oceanial, People with dual nationality



OUTPUT: O_master_data_key             pd of final clean data with the correct 
                                      feature collection
      [.csv file in current directory of final data filename is I_out_filename]

Example: 
I_data_path = 'C:/Users/Victor.SalasAranda/Desktop/FSA/csv/UnifiedView_NOMIS_CDRC_Seasonality.csv'
I_keyarray = [1,1,1,1,1,1,1,1]
I_out_filename = 'master'
data = Master_options(I_data_path, I_keyarray, I_out_filename)
data.head()

RUN:    TBD
         
         
@author: Victor
Created on Wed 2020/01/24
modified: 
    
"""

    # In[A]:

def Master_options(I_data_path, I_keyarray, I_out_filename): 
    
    #Load libraries
    import pandas as pd
    import os
    import warnings
    warnings.filterwarnings('ignore') 
            
    # In[B]:
    # Define inputs      

    Master_path = I_data_path     # './UnifiedView_NOMIS_CDRC_Seasonality.csv' - run from previous function
    o_fname = I_out_filename;

    key_age,                key_CountryBirth = I_keyarray[0], I_keyarray[1]
    key_economicActivity,   key_health       = I_keyarray[2], I_keyarray[3]
    key_hoursWorked,        key_Language     = I_keyarray[4], I_keyarray[5]
    key_Occupation,         key_Passport     = I_keyarray[6], I_keyarray[7]
    
    print('Keys selected: \n  key_age: '+ str(key_age) + ', key_CountryBirth: ' + str(key_CountryBirth)+
         ', key_economicActivity: '+ str(key_economicActivity) + ', key_health: ' + str(key_health)+
         ', \n key_hoursWorked: '+ str(key_hoursWorked) + ', key_Language: ' + str(key_Language)+
         ', key_Occupation: '+ str(key_Occupation) + ', key_Passport: ' + str(key_Passport))
    # In[2]:      
    #Load data
    
    data = pd.read_csv(Master_path)
    del data['Unnamed: 0']
    
    print('Imported dataset has:',data.shape[0],'rows and', data.shape[1], 'columns')
    
    def count_extract(column_name, original_df):
        OUTCOME_CATEGORIES = [0, 1, 2, 3, 4, 5]
        df = pd.DataFrame(original_df[column_name].value_counts())
        df.sort_index(level=OUTCOME_CATEGORIES, inplace=True) 
        df.columns = [column_name]
        l = []
        count = 0
        for i in range(len(df)):
            count+=df[column_name][i]
        for j in range(len(df)):
            l.append(str(round(df[column_name][j]/count*100,2))+ '%')
        df['Percentage'] = l
        df.sort_index(level=OUTCOME_CATEGORIES, inplace=True)            
        display(df)
        del df
    
    #print('FSA Rating distribution:')
    #count_extract('summaryScore', data)
    # In[4]:
    #New dataframe

    data2 = data.copy()
    print('Selecting features by each key...')
        
    # In[8]:
        
    ##Age Structure
    
    FirstColumn = 'Age: All usual residents; measures: Value'
    LastColumn = 'Age: Median Age; measures: Value'
    a = data2.columns.get_loc(FirstColumn)
    b = data2.columns.get_loc(LastColumn)
    tempfeature = data2.iloc[:, a:b+1]
    
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = data2.drop(tempfeature, axis = 1)
    
    ## KEY
    # 1: C0 = All ages
    # 2: New columns: 0 to 14, 15 to 19, 20 to 29, 30 to 44, 45 to 59, 60 to 74 and 75 and over
    # 3: C1,...,C16 = Age0_4,...,Age90+
    
    #CHANGE KEY HERE!!!! 
    #key_age = 2
    
    if key_age == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, 0]) 
    if key_age == 2:    
        #Summing up ages between 0 to 14 
        tempfeature['New Age: Age 0 to 14'] = tempfeature.iloc[:, 1:5].sum(axis = 1)
        cols = list(tempfeature)
        cols.insert(len(cols)-1, cols.pop(cols.index('New Age: Age 0 to 14')))
        tempfeature = tempfeature.ix[:, cols]

        #Summing up ages between 15 to 19 
        tempfeature['New Age: Age 15 to 19'] = tempfeature.iloc[:, 5:8].sum(axis = 1)
        cols = list(tempfeature)
        cols.insert(len(cols)-1, cols.pop(cols.index('New Age: Age 15 to 19')))
        tempfeature = tempfeature.ix[:, cols]

        #Summing up ages between 20 to 29 
        tempfeature['New Age: Age 20 to 29'] = tempfeature.iloc[:, 8:10].sum(axis = 1)
        cols = list(tempfeature)
        cols.insert(len(cols)-1, cols.pop(cols.index('New Age: Age 20 to 29')))
        tempfeature = tempfeature.ix[:, cols]

        #Summing up ages between 30 to 44 
        tempfeature['New Age: Age 30 to 44'] = tempfeature.iloc[:, 10:11].sum(axis = 1)
        cols = list(tempfeature)
        cols.insert(len(cols)-1, cols.pop(cols.index('New Age: Age 30 to 44')))
        tempfeature = tempfeature.ix[:, cols]

        #Summing up ages between 45 to 59 
        tempfeature['New Age: Age 45 to 59'] = tempfeature.iloc[:, 11:12].sum(axis = 1)
        cols = list(tempfeature)
        cols.insert(len(cols)-1, cols.pop(cols.index('New Age: Age 45 to 59')))
        tempfeature = tempfeature.ix[:, cols]

        #Summing up ages between 60 to 74
        tempfeature['New Age: Age 60 to 74'] = tempfeature.iloc[:, 12:14].sum(axis = 1)
        cols = list(tempfeature)
        cols.insert(len(cols)-1, cols.pop(cols.index('New Age: Age 60 to 74')))
        tempfeature = tempfeature.ix[:, cols]

        #Summing up ages between 75 and over
        tempfeature['New Age: Age 75 and over'] = tempfeature.iloc[:, 14:17].sum(axis = 1)
        cols = list(tempfeature)
        cols.insert(len(cols)-1, cols.pop(cols.index('New Age: Age 75 and over')))
        tempfeature = tempfeature.ix[:, cols]

        feature = pd.DataFrame(tempfeature.iloc[:, tempfeature.shape[1]-7:tempfeature.shape[1]+1])

    elif key_age == 3:
        feature = pd.DataFrame(tempfeature.iloc[:, 1:17])
    
    #Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    
    #Copy to a new dataframe
    tempsubmasterAge = tempsubmaster
   
    # In[9]:
    
    ## Country of Birth

    ##Clear any existing df
    del tempfeature
    del tempsubmaster
    del feature
    
    FirstColumn = 'Country of Birth: All usual residents; measures: Value'
    LastColumn = 'Country of Birth: Other countries; measures: Value'
    a = tempsubmasterAge.columns.get_loc(FirstColumn)
    b = tempsubmasterAge.columns.get_loc(LastColumn)
    tempfeature = tempsubmasterAge.iloc[:, a:b+1]
    
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = tempsubmasterAge.drop(tempfeature, axis = 1)
    
    ##To create a new feature: Europe + Ireland
    tempfeature['RealEurope'] = pd.DataFrame(tempfeature.iloc[:, 7:9]).sum(axis=1)
    
    ## KEY
    # 1: C0 = All
    # 2: C1,C7, C8, C11 = UK, Ireland, EU, RoW
    # 3: C1, C7+C8 (=C12), C11 = UK, Ral EU, RoW
    # 4: C1,C7,C9,C10,C11 = UK, Ireland, EU2001-, EU2001+, RoW
    # 5: C2,...,C7 and C9,...,C11 = England,..., Wales and EU2001-, EU2001+, RoW
    
    #CHANGE KEY HERE!!!! 
    #key_CountryBirth = 3
    
    if key_CountryBirth == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, 0]) 
    elif key_CountryBirth == 2:
        feature = pd.DataFrame(tempfeature.iloc[:, [1,7,8,11]])
    elif key_CountryBirth == 3:
        feature = pd.DataFrame(tempfeature.iloc[:, [1,11,12]])
    elif key_CountryBirth == 4:
        feature = pd.DataFrame(tempfeature.iloc[:, [1,7,9,10,11]])
    elif key_CountryBirth == 5:
        feature = pd.DataFrame(tempfeature.iloc[:, list(range(2,8)) + list(range(9,12))])
      
    #Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    tempsubmasterBirth = tempsubmaster

    
    # In[11]:
    
    ## Economic Activity by Sex
    
    #Clear any existing df
    del tempfeature
    del tempsubmaster
    del feature
    
    FirstColumn = 'Sex: All persons; Economic Activity: All usual residents aged 16 to 74; measures: Value'
    LastColumn = 'Sex: Females; Economic Activity: Long-term unemployed; measures: Value'
    a = tempsubmasterBirth.columns.get_loc(FirstColumn)
    b = tempsubmasterBirth.columns.get_loc(LastColumn)
    tempfeature = tempsubmasterBirth.iloc[:, a:b+1]
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = tempsubmasterBirth.drop(tempfeature, axis = 1)
    
    ## KEY
    # 1: C0 = Economic Activity's All persons
    # 2: C18 and C36 = Economic Activity by Male and Female
    # 3: C2, C6+C7 and C8 = Total empl. + not empl + inact. total
    # 4: C20, C24+C25, C26, C42+C43 and C44 = Total empl. + not empl + inact. total by Male and Female
    # 5: C3,...,C7 and C9,...,C13 = All subcat minus total average column
    # 6: C21,...,C25, C27,...,C31, C39,...,43 and C45,...,C49 = All subcat minus total average column by Male and Female
    
    #CHANGE KEY HERE!!!! 
    #key_economicActivity = 2
    
    if key_economicActivity == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, 0]) #Column[1]
    elif key_economicActivity == 2:
        feature = pd.DataFrame(tempfeature.iloc[:, [18, 36]]) #Column[19] & Column[37]
    elif key_economicActivity == 3:
        feature = pd.DataFrame(tempfeature.iloc[:, [3,8]])
        feature['Sex: All persons; Economic Activity: Economically active: Not Employed'] = pd.DataFrame(tempfeature.iloc[:, 6:8].sum(axis = 1)) #Column[2], Sum(Column[6], Column[7]), Column[8]
    elif key_economicActivity == 4:
        feature = pd.DataFrame(tempfeature.iloc[:, [20, 25, 26, 37, 44]]) #columns[20], Sum(columns[24], column[25]), column[26], columns[37], Sum(columns[42], column[43]), column[44] 
        feature['Sex: Males; Economic Activity: Economically active: Not Employed'] = pd.DataFrame(tempfeature.iloc[:, 24:26].sum())
        feature['Sex: Females; Economic Activity: Economically active: Not Employed'] = pd.DataFrame(tempfeature.iloc[:, 42:44].sum())
    elif key_economicActivity == 5:
        feature = pd.DataFrame(tempfeature.iloc[:, list(range(3,8)) + list(range(9,14))]) #columns[3 to 7] & columns[9 to 13]
    elif key_economicActivity == 6:
        feature = pd.DataFrame(tempfeature.iloc[:, list(range(21,26)) + list(range(27,32)) + list(range(39,44)) +list(range(45,50))]) #columns[21 to 25], columns[27 to 31], columns[39 to 43], columns[45, 49]
    
    #Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    tempsubmasterSex = tempsubmaster
    
    # In[13]:
    
    ## Health and Provision of Unpaid Care

    ## Clear any existing df
    del tempfeature
    del tempsubmaster
    del feature

    FirstColumn = 'disability/health/care: All categories: Long-term health problem or disability; measures: Value'
    LastColumn = 'disability/health/care: Provides 50 or more hours unpaid care a week; measures: Value'
    a = tempsubmasterSex.columns.get_loc(FirstColumn)
    b = tempsubmasterSex.columns.get_loc(LastColumn)
    tempfeature = tempsubmasterSex.iloc[:, a:b+1]
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = tempsubmasterSex.drop(tempfeature, axis = 1)
    
    ## KEY
    # 1: C0 = All
    # 2: C1,...,C3 = limited a lot, little and not limited
    # 3: C7,...,C11 = very good, good, fair, bad and very bad health
    # 4: C12,...,C15 = Provides no unpaid care, Provides 1 to 19 hours/week,...,Provides 50 or more hours a week
    # 5: C4,..,C6 and C4_star,...,C6_star  = limited a lot, little and not limited by Age  (Age 16 to 64 and Age 0 to 15/Age 65+)
    #      with (e.g) C4_star = C1 - C4
    
    #CHANGE KEY HERE!!!! 
    #key_health = 5
    
    if key_health == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, 0]) #Column 0
    elif key_health == 2:
        feature = pd.DataFrame(tempfeature.iloc[:, 1:4]) #Columns 1 to 3
    elif key_health == 3:
        feature = pd.DataFrame(tempfeature.iloc[:, 7:12]) #Columns 7 to 11
    elif key_health == 4:
        feature = pd.DataFrame(tempfeature.iloc[:, 12:16]) #Columns 12 to 15
    elif key_health == 5:
        feature = pd.DataFrame(tempfeature.iloc[:, 4:7]) #Columns 4 to 6
        feature['disability/health/care: Day-to-day activities limited a lot: Age 0 to 15/Age 65+'] = pd.DataFrame(tempfeature.iloc[:, 1] - tempfeature.iloc[:,4])
        feature['disability/health/care: Day-to-day activities limited a little: Age 0 to 15/Age 65+'] = pd.DataFrame(tempfeature.iloc[:, 2] - tempfeature.iloc[:,5])
        feature['disability/health/care: Day-to-day activities not limited: Age 0 to 15/Age 65+'] = pd.DataFrame(tempfeature.iloc[:, 3] - tempfeature.iloc[:,6])
        
        #Column 0 minus Columns[4 to 6]
        
    #Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    tempsubmasterHealth = tempsubmaster
    
    # In[15]:
       
    ## Hours Worked 

    #Clear any existing df
    del tempfeature
    del tempsubmaster
    del feature
    
    FirstColumn = 'Hours Worked: All usual residents aged 16 to 74 in employment the week before the census; measures: Value'
    LastColumn = 'Hours Worked: Females: Full-time: 49 or more hours worked; measures: Value'
    a = tempsubmasterHealth.columns.get_loc(FirstColumn)
    b = tempsubmasterHealth.columns.get_loc(LastColumn)
    tempfeature = tempsubmasterHealth.iloc[:, a:b+1]
    
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = tempsubmasterHealth.drop(tempfeature, axis = 1)
    
    ## KEY
    # 1: C0 = All
    # 2: C1,...,C4 = Part-time: 15 hours,...,49 or more hours
    # 3: C5 and C10 = All by Male and Female
    # 4: C6,...,C9 and C11,...,C14 = Part-time: 15 hours,...,49 or more hours by Male and Female
    
    
    #CHANGE KEY HERE!!!! 
    #key_hoursWorked = 3
    
    if key_hoursWorked == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, 0]) 
    elif key_hoursWorked == 2:
        feature = pd.DataFrame(tempfeature.iloc[:, 1:5])
    elif key_hoursWorked == 3:
        feature = pd.DataFrame(tempfeature.iloc[:, [5,10]])
    elif key_hoursWorked == 4:
        feature = pd.DataFrame(tempfeature.iloc[:, list(range(6,10)) + list(range(11,15))])
    
        
    #Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    tempsubmasterHours = tempsubmaster
    
    # In[17]:
    
    ## Household Language

    ##Clear any existing df
    del tempfeature
    del tempsubmaster
    del feature
 
    FirstColumn = 'Main Language: All categories: English as a household language; measures: Value'
    LastColumn = 'Main Language: No people in household have English as a main language (English or Welsh in Wales); measures: Value'
    a = tempsubmasterHours.columns.get_loc(FirstColumn)
    b = tempsubmasterHours.columns.get_loc(LastColumn)
    tempfeature = tempsubmasterHours.iloc[:, a:b+1]
    
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = tempsubmasterHours.drop(tempfeature, axis = 1)
    
    ## KEY
    # 1: C0 = All
    # 2: C1,...,C4 = All people aged 16 and over in household have English as a main language ,...,No people in household have English as a main language
    
    
    #CHANGE KEY HERE!!!! 
    #key_Language = 2
    
    if key_Language == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, 0]) 
    elif key_Language == 2:
        feature = pd.DataFrame(tempfeature.iloc[:, 1:5])
        
    ## Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    tempsubmasterHousehold = tempsubmaster
    
    # In[19]:

    ## Household Language
    
    ## Clear any existing df
    del tempfeature
    del tempsubmaster
    del feature
    
    FirstColumn = 'Sex: All persons; Occupation: All categories: Occupation; measures: Value'
    LastColumn = 'Sex: Females; Occupation: 9. Elementary occupations; measures: Value'
    a = tempsubmasterHousehold.columns.get_loc(FirstColumn)
    b = tempsubmasterHousehold.columns.get_loc(LastColumn)
    tempfeature = tempsubmasterHousehold.iloc[:, a:b+1]
    
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = tempsubmasterHousehold.drop(tempfeature, axis = 1)
    
    ## KEY
    # 1: C0 = All 
    # 2: C1,...,C9 = Managers, directors and senior officials,..., Elementary occupations
    # 3: C10 and C20 = All by Male and Female
    # 4: C11,...,C19 and C21,...,C29 = Managers, directors and senior officials,..., Elementary occupations by Male and Female
    
    
    #CHANGE KEY HERE!!!! 
    #key_Occupation = 3
    
    if key_Occupation == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, 0]) 
    elif key_Occupation == 2:
        feature = pd.DataFrame(tempfeature.iloc[:, 1:10])
    elif key_Occupation == 3:
        feature = pd.DataFrame(tempfeature.iloc[:, [10,20]])
    elif key_Occupation == 4:
        feature = pd.DataFrame(tempfeature.iloc[:, list(range(11,20)) + list(range(21,30))])
    
    
    
    #Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    tempsubmasterOccupation = tempsubmaster

    
    # In[21]:
    
    ##Occupation by Sex
    
    ##Clear any existing df
    del tempfeature
    del tempsubmaster
    del feature
    

    FirstColumn = 'Passports Held: All usual residents; measures: Value'
    LastColumn = 'Passports Held: Antarctica and Oceania; measures: Value'
    a = tempsubmasterOccupation.columns.get_loc(FirstColumn)
    b = tempsubmasterOccupation.columns.get_loc(LastColumn)
    tempfeature = tempsubmasterOccupation.iloc[:, a:b+1]
    
    ##Remove tempfeature from tempsubmaster
    tempsubmaster = tempsubmasterOccupation.drop(tempfeature, axis = 1)
    
    #To create the new column: People with dual nationality
    tempfeature['a'] = pd.DataFrame(tempfeature.iloc[:, 1:12]).sum(axis=1)
    tempfeature['People with dual nationality'] = tempfeature.apply(lambda row: row.a - row['Passports Held: All usual residents; measures: Value'], axis=1)
    del tempfeature['a']
    
    ## KEY
    # 1: C0 and C12 = All, People with dual nationality
    # 2: C1,...,C11 and C12 =  No passport,..., passport from Antarctica and Oceanial, People with dual nationality
     
        
    #CHANGE KEY HERE!!!! 
    #key_Passport = 1
    
    if key_Passport == 1:
        feature = pd.DataFrame(tempfeature.iloc[:, [0,12]])
    elif key_Passport == 2:
        feature = pd.DataFrame(tempfeature.iloc[:, 1:13])
    
    
    
    #Append tempsubmaster and tempfeature
    tempsubmaster = pd.concat([tempsubmaster, feature], axis = 1)
    tempsubmasterPassport = tempsubmaster # In case you want to add new features
    
    
    # In[25]:
    ### Export Cleaned Unified View Data:  
    
    key = str(key_age)+'_'+ str(key_CountryBirth)+'_'+ str(key_economicActivity)+'_'+ str(key_health)+'_'+ str(key_hoursWorked)+'_'+str(key_Language)+'_'+str(key_Occupation)+'_'+str(key_Passport)
    print('Completed')
    
    #print('Option dataset:')
    #display(tempsubmaster.head())
    print('Working dataset has: ' +  str(tempsubmaster.shape[0])+ ' rows and ' + str(tempsubmaster.shape[1])+ ' columns')
    ## Save as csv 
    os.makedirs('./csv', exist_ok=True)
    out_fpath = './csv/'
    out_fpath_name = out_fpath + o_fname + '_'+ key + '.csv'
    print('Saving dataset to '+str(out_fpath_name))
    

    tempsubmaster.to_csv(out_fpath_name)
 
    O_master_data_key = tempsubmaster  
    
    ################################################END#################################################
    return key, O_master_data_key
