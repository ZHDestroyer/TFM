#!/usr/bin/env python
# coding: utf-8

# # Unified View Master (Cleaning)

"""
UnifiedView_MasterClean:
    function file to create a clean UV data set and output in pd form and .csv 
    Program came from Aishat UnifiedView_Master_script_function.ipynb within 
    [github]/main/Funtions 
        
INPUT:  I_uv_path         stg. file path and name of the UV raw data 
        I_ukpostcode      stg. file path of dictionary for postcodes and lat/lon
        I_out_filename    stg. filename only of output csv for clean UV data 

OUTPUT: O_uv_clean        pd of clean UV data 
        [.csv file in current directory of clean UV data]

RUN: 
         rawUVpath = 'C:/Users/Neel.Savani-Patel/NEEL_tech/LOCAL_DATA/_business_data_fhrs-latest-inspection__view_detail.csv'
         ukpostcode = 'C:/Users/Neel.Savani-Patel/NEEL_tech/LOCAL_DATA/ukpostcodes.csv'
         uv_clean_filename = 'UnifiedView_Master.csv'
         pd_UV_clean = UnifiedView_cleaning (rawUVpath, ukpostcode, uv_clean_filename)
         
         
@author:NPS
Created on Wed 2020/01/14
modified: VSA Wed 2020/02/5
    
"""


def UnifiedView_cleaning(I_uv_path, I_ukpostcode, I_out_filename):
    
    #Load libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import norm
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    import warnings
    
    warnings.filterwarnings('ignore') 
    get_ipython().run_line_magic('matplotlib', 'inline')
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    
    
    
    #%% B. define inputs    
    uv_path = I_uv_path          # "Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/_business_data_fhrs-latest-inspection__view_detail.csv"
    ukpost_path = I_ukpostcode   # "Q:/Inform/Strategic Surveillance/Sprint AI/DATA/ukpostcodes.csv"
    o_fname = I_out_filename;
    
    #%% 1. load in .csv file as a pd
    uni_view = pd.read_csv(uv_path)

    ##BACKUP UV raw data is this!
    #uni_view = pd.read_csv("Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/_business_data_fhrs-latest-inspection__view_detail.csv")

    
    #%% 2. read pd and Extract Postcodes from Full Address Line
    
    
    ### Unified View: Data Cleaning
    ##Extract Postcodes from Full Address Line
    ##Convert 'establishment.premises.prefAddress.extendedAddress' to string 
    uni_view['establishment.premises.prefAddress.extendedAddress'] = uni_view['establishment.premises.prefAddress.extendedAddress'].astype('str')

    ##Extract postcode from address columns 'establishment.premises.prefAddress.extendedAddress'
    l = []
    for i in range(len(uni_view)):
       m = len(uni_view['establishment.premises.prefAddress.extendedAddress'][i])
       if uni_view['establishment.premises.prefAddress.extendedAddress'][i][m-1].isdigit()==False:
           if uni_view['establishment.premises.prefAddress.extendedAddress'][i][m-2].isdigit()==False:
               if uni_view['establishment.premises.prefAddress.extendedAddress'][i][m-3].isdigit()==True: 
                   if uni_view['establishment.premises.prefAddress.extendedAddress'][i][m-8] == ' ':
                       l.append(uni_view['establishment.premises.prefAddress.extendedAddress'][i][m-7:m])
                   else:
                       l.append(uni_view['establishment.premises.prefAddress.extendedAddress'][i][m-8:m])
               else:
                   l.append(np.nan)
           else:
               l.append(np.nan)
       else:
           l.append(np.nan)
    uni_view['Postcode2'] = l

    #%% 3. Select Rows in Latitude, Longitude & Postcode without NaN Values
    
    len(uni_view) ##996482
    A = uni_view['establishment.premises.prefAddress.lat'].isnull() ##339292  =len(A)
    B = uni_view['establishment.premises.prefAddress.long'].isnull() ##339292
    C1 = uni_view['establishment.premises.prefAddress.extendedAddress'].isnull() ##317
    C2 = uni_view['Postcode2'].isnull() ##242450
    D1 = A &(B)&(C1) ##317
    D2 = A &(B)&(C2) ##242178
    E1 = D1==False ##996165   = ##Number rows without lot&lat and Postcode's NAs
    E2 = D2==False ##754304 = len(E) = Number rows without lot&lat and Postcode's NAs + without 'private address'
    F = uni_view['establishment.establishmentRegistration.authorityEstablishmentID'].isnull() ##0
    G = len(uni_view['establishment.establishmentRegistration.authorityEstablishmentID'].unique().tolist()) ##586533 (number of unique ID)
    uni_view = uni_view[E2] ##select the rows without lot&lat and Postcode's NAs

    uni_view['establishment.premises.prefAddress.long' ] = uni_view['establishment.premises.prefAddress.long'].astype(str)
    uni_view['establishment.premises.prefAddress.lat'] = uni_view['establishment.premises.prefAddress.lat'].astype(str)
    a = [list(range(0,len(uni_view)))]
    uni_view = uni_view.set_index(a)


    #%% 4. IMPORT POSTCODE DICTIONARY 
    
    ## Filling the Latitude & Longitude NaNs 
    ## Load UK postcodes table for look up
    ukpostcodes = pd.read_csv(ukpost_path)

    #%% 5. fix gaps in postcodes

    ##Merge Unified View with UK Lookup Table
    uni_view = pd.merge(uni_view, ukpostcodes[['postcode', 'latitude','longitude']],
                        left_on = "Postcode2", right_on = 'postcode', how = "left")

    ##Convert to string 
    uni_view['establishment.premises.prefAddress.long'] = uni_view['establishment.premises.prefAddress.long'].astype(str)
    uni_view['establishment.premises.prefAddress.lat'] = uni_view['establishment.premises.prefAddress.lat'].astype(str)

    ##Replacing longitude and latitude NaN values 
    for i in range(len(uni_view)):
        if uni_view['establishment.premises.prefAddress.lat'][i] == 'nan':
            uni_view['establishment.premises.prefAddress.lat'][i] = uni_view['latitude'][i]
        if uni_view['establishment.premises.prefAddress.long'][i] == 'nan':
            uni_view['establishment.premises.prefAddress.long'][i] = uni_view['longitude'][i]


    #%% 6. delete duplicates from the 
    ##Deleting Duplicated & Triplicated Values
    ##Convert to string
    uni_view['establishment.premises.prefAddress.lat'] = uni_view['establishment.premises.prefAddress.lat'].astype(str)
    uni_view['establishment.premises.prefAddress.long'] = uni_view['establishment.premises.prefAddress.long'].astype(str)

    a = [list(range(0, len(uni_view)))]
    uni_view = uni_view.set_index(a)

    for i in range(len(uni_view)):
        n = len(uni_view['establishment.premises.prefAddress.long'][i])
        m = len(uni_view['establishment.premises.prefAddress.lat'][i])
        if '|' in uni_view['establishment.premises.prefAddress.long'][i]: #for duplicated and triplicated values
            k1 = uni_view['establishment.premises.prefAddress.long'][i].index('|')
            k2 = uni_view['establishment.premises.prefAddress.lat'][i].index('|')
            uni_view['establishment.premises.prefAddress.long'][i] = uni_view['establishment.premises.prefAddress.long'][i][:k1] 
            uni_view['establishment.premises.prefAddress.lat'][i] = uni_view['establishment.premises.prefAddress.lat'][i][:k2] 


    ##Extract non-null rows in establishment.premises.prefAddress.lat 
    A = uni_view['establishment.premises.prefAddress.lat'].notnull() 
    ##len(uni_view[A]) #744684
    uni_view = uni_view[A] # I have deleted: 754304 -744684 = 9620 because the PC are depricated


    ##Drop columns not needed
    uni_view = uni_view.drop(columns = ['@id', 'type', 'establishment', 'establishment.tradingName', 
                                        'establishment.establishmentRN', 'rating', 
                                        'rating.notation', 'inspectionScheme', 'establishment.premises.prefAddress.uprn', 
                                        'context.operatorName', 'lodgingDateTime', 'determinationDateTime', 
                                        'originalInspectionDateTime', 'hasEnd'])


    ##Remove commas
    uni_view['Postcode2'] = uni_view['Postcode2'].str.replace(',', '')
    ##Remove whitespace on left-hand-side
    uni_view["Postcode2"] = uni_view["Postcode2"].str.lstrip()

    ##Create postcode column (XXX Y) by removing last two characters
    uni_view["Postcode_Sector"] = uni_view["Postcode2"].astype(str).str[:-2]


    #%% 7. remove and refill NaN values 
    ##Removing NaN Values from Unified View
    uni_view = uni_view.dropna(subset = ['context.tradingName', 'summaryScore', 'hygieneScore', 
                                         'structuralScore', 'confidenceScore', 'publicationDateTime', 
                                         'Postcode2', 'latitude', 'longitude', 'Postcode_Sector'], axis = 0)


    ##Fill NAN - column with {nan, True} -> {False, True}
    uni_view['isAppeal'] = uni_view['isAppeal'].fillna('none')
    uni_view['appealStatus.notation'] = uni_view['appealStatus.notation'].fillna('none')
    uni_view['exercisedRightToReply'] = uni_view['exercisedRightToReply'].fillna(False)
    uni_view['isOnHold'] = uni_view['isOnHold'].fillna(False)
    uni_view['isRescore'] = uni_view['isRescore'].fillna(False)
    uni_view['originalSummaryScore'] = uni_view['originalSummaryScore'].fillna('none')
    uni_view['inspectionPriorToSchemeStartDate'] = uni_view['inspectionPriorToSchemeStartDate'].fillna(False)
    uni_view['inspectionProcessStatus.label'] = uni_view['inspectionProcessStatus.label'].fillna('none')
    uni_view['inspectionProcessStatus.notation'] = uni_view['inspectionProcessStatus.notation'].fillna('none')




    #%% 40. return ouputs

    ### Export Cleaned Unified View Data
    #out_filpath = r"Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/"
    out_fpath = './'
    out_fpath_name = out_fpath + o_fname
    
    uni_view.to_csv(out_fpath_name)
    #############################################END####################################################

    return uni_view
