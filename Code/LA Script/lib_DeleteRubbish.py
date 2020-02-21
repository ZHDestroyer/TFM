# -*- coding: utf-8 -*-



"""
Master_feature_engineer:
    -Function file to quickly delete some of the features we don't need or which are useless

I_csvfilename_of_features, I_pd_Masterlist, I_out_filename

INPUT:  I_pd_Masterlist                         stg. file path and name of final merged master data  
        I_csvfilename_of_features               stg. file path and name of deleting feature csv 
        I_out_filename                          stg. filename only of output csv for UV master data 




OUTPUT: clean_master_data             pd of final clean data with the correct 
                                      feature collection
      [.csv file in current directory of final data filename is I_out_filename]

Example: 
I_pd_Masterlist = 'C:/Users/Victor.SalasAranda/Desktop/Github/Sprint-AI-01-FHRS/victor/Engineer/master_1_1_1_1_1_1_1_1.csv'
I_csvfilename_of_features = 'C:/Users/Victor.SalasAranda/Desktop/Github/Sprint-AI-01-FHRS/victor/Engineer/Feature Deletion - 1_1_1_1_1_1_1.csv'
I_out_filename = 'clean_master'
data = delete_rubbish(I_pd_Masterlist, I_csvfilename_of_features, I_out_filename)
data.head()

RUN:    TBD
         
         
@author: VSA
Created on Thu 2020/01/30
modified:  
    
"""

    # In[A]:

def delete_rubbish(I_pd_Masterlist, I_csvfilename_of_features, I_out_filename, Bool):
    
    #Load libraries
    import pandas as pd
    import os
    import warnings
    warnings.filterwarnings('ignore') 
    
    # Define inputs
    Master_path = I_pd_Masterlist     
    deleletepd = I_csvfilename_of_features
    o_fname = I_out_filename;
        
    # Load Data
    data = pd.read_csv(Master_path)
    del data['Unnamed: 0']
             
    #Solving character issues
    if 'Working age population 18-59/64: for use with Employment Deprivation Domain (excluding prisoners) ' in data.columns.tolist():
        data = data.rename(columns={'Working age population 18-59/64: for use with Employment Deprivation Domain (excluding prisoners) ':'Working age population 18-59/64: for use with Employment Deprivation Domain (excluding prisoners)'})
    
    #Load files
    datadelete = pd.read_csv(deleletepd)
    print('There are ' +  str(datadelete.shape[0])+ ' different features to remove')
    if Bool == 'Yes':
        display(datadelete)
    deletelist = datadelete['Columns to Delete'].tolist()
    
    for i in range(len(deletelist)):
        if deletelist[i] in data.columns.tolist():
            del data[deletelist[i]]
        else:
            print("This feature doesn't exist: " + str(deletelist[i]) + " please, rewrite it correctly")
    
    #print('Final dataset:')
    #display(data.head())
    print('Working dataset has: ' +  str(data.shape[0])+ ' rows and ' + str(data.shape[1])+ ' columns')

    ## Save as csv 
    os.makedirs('./csv', exist_ok=True)
    out_fpath = './csv/'
    out_fpath_name = out_fpath + o_fname  + '.csv' #Master_path[len(Master_path)-26:] + _ 
    print('Saving dataset to '+str(out_fpath_name))
    data.to_csv(out_fpath_name)

    clean_master_data = data
    return clean_master_data


