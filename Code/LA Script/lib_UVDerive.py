#!/usr/bin/env python
# coding: utf-8

# # Adding Seasonality columns to Unified View

"""
add_UnifiedView_Seasonality:
    -Function file to seasonality columns to pre cleaned, prenomis merge UV & cdrc merge data 
    saved as .csv
    -Program came from Aishat UnifiedView_NOMISfeatures_script_function.ipynb within 
    [github]/main/Funtions.
    -Program requires:
        1. Adding seasonality columns to UnifiedView_NOMIS_CDRC.csv using function - uv_add_seasonality

INPUT:  I_uvNomisCdrc_path      stg. file path and name of UV+nomis+cdrc merged data  
        I_out_filename          stg. filename only of output csv for clean UV data 

OUTPUT: O_uv_seasonality          pd of clean UV data 
        [.csv file in current directory of UV_seasonality added data]

RUN: 
            UV_cdrc_Master = './UnifiedView_NOMIS_CDRC.csv'
            uvseasonality_filename = 'UV_seasonality_Master.csv'  
            pd_UV_seasonality = uv_seasonality.uv_add_seasonality(UV_cdrc_Master, uvseasonality_filename)
         
         
@author:AO
Created on Thurs 2020/01/15
modified: 

"""



"""
SEASONALITY of inspection
 
Must be run after uv_merge_cdrc function in uv cleaning
 
"""
def uv_seasonality(I_uvNomisCdrc_path, I_out_filename):
    #Load libraries 
    import pandas as pd
    import numpy as np
    import warnings
    from datetime import datetime
    
    #%% B. define inputs      
    uv_path = I_uvNomisCdrc_path     # './UnifiedView_NOMIS_CDRC.csv' - run from previous function
    o_fname = I_out_filename;

    #%% 1. load in .csv file as a pd  UnifiedView_NOMIS_CDRC
    
    UnifiedView_NOMIS_CDRC = pd.read_csv(uv_path)
    UnifiedView_NOMIS_CDRC = UnifiedView_NOMIS_CDRC.drop(columns = 'Unnamed: 0')

    #%% 2. Create Seasonality dataset
    uv_Season = UnifiedView_NOMIS_CDRC.copy()
    
    #%% 3. Change date column to datetime type
    uv_Season['inspectionDateTime'] = pd.to_datetime(uv_Season['inspectionDateTime'])


    #%% 4. Split date column into weekday, day, month and year
    uv_Season['weekdayOfInspection'] = uv_Season['inspectionDateTime'].dt.day_name()
    uv_Season['dayOfInspection']     = uv_Season['inspectionDateTime'].dt.day
    uv_Season['monthOfInspection']   = uv_Season['inspectionDateTime'].dt.month
    uv_Season['yearOfInspection']    = uv_Season['inspectionDateTime'].dt.year

    #%% 5. Sort columns
    #cols = list(uv_Season)
    #cols.insert(12, cols.pop(cols.index('weekdayOfInspection')))
    #cols.insert(13, cols.pop(cols.index('dayOfInspection')))
    #cols.insert(14, cols.pop(cols.index('monthOfInspection')))
    #cols.insert(15, cols.pop(cols.index('yearOfInspection')))
    #uv_Season = uv_Season.ix[:, cols]
    
    ### Export Cleaned Unified View Data
    #out_filpath = r"Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/"
    out_fpath = './'
    out_fpath_name = o_fname

    uv_Season.to_csv(out_fpath_name)
    ##out_filpath
    #uv_Season.to_csv(r"Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/uv_Season.csv")
    
    #pickle of uv_Season
    ################################################END#################################################
    return uv_Season
    


  
"""
CHAIN RESTURANT.. large chains often have standard hygien policies to follow

Must be run after uv_seasonality function above

"""

def uv_chain(I_uvmerge_path, I_out_filename):
    
    
    #Load libraries 
    import pandas as pd
    import numpy as np
    
    #%% B. define inputs      
    uv_path = I_uvmerge_path     # './UnifiedView_NOMIS_CDRC_Seasonality.csv' - run from previous function
    o_fname = I_out_filename;    # filename of output data strucutre in csv

    #%% 1. load in .csv file as a pd  UnifiedView_NOMIS_CDRC_Seasonality
    UnifiedView_NOMIS_CDRC_Seasonality = pd.read_csv(uv_path)
    
    #Delete unwanted column(s)
    UnifiedView_NOMIS_CDRC_Seasonality = UnifiedView_NOMIS_CDRC_Seasonality.drop(columns = 'Unnamed: 0')
    
    #Copy into new df
    UV_NOMIS_CDRC_Seasonality_Chain = UnifiedView_NOMIS_CDRC_Seasonality.copy()

    #%% 2. Create frequency data set of FBO names
    ##Number of times restaurant name appears in columm
    chain = pd.DataFrame(UV_NOMIS_CDRC_Seasonality_Chain['context.tradingName'].value_counts())
    ##Reset index and rename columns 
    chain = chain.reset_index()
    chain.columns = ['context.tradingName', 'count']

    
    #%% 3. Map frequency to master dataset
    mapping = dict(chain[['context.tradingName', 'count']].values)
    UV_NOMIS_CDRC_Seasonality_Chain['ChainCount'] = UV_NOMIS_CDRC_Seasonality_Chain['context.tradingName'].map(mapping)

    #%% 4. Decide whether FBO is a chain or not
    UV_NOMIS_CDRC_Seasonality_Chain['ChainOrNot'] = np.where(
    UV_NOMIS_CDRC_Seasonality_Chain['ChainCount'] > 1, 'yes', 'no')
    
    ### Export Cleaned Unified View Data
    out_fpath = './'
    out_fpath_name = out_fpath + o_fname
    
    UV_NOMIS_CDRC_Seasonality_Chain.to_csv(out_fpath_name)
    ##out_filpath
    #UV_NOMIS_CDRC_Seasonality_Chain.to_csv(r"Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/UV_NOMIS_CDRC_Seasonality_Chain.csv")
    ################################################END#################################################
    return UV_NOMIS_CDRC_Seasonality_Chain
    





