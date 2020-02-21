#!/usr/bin/env python
# coding: utf-8

# # Unified View + NOMIS Features & CDRC Features - Merger

"""
merge_UnifiedView_CDRC:
    -function file to merge CDRC data into pre cleaned and prenomis merge UV 
    data saved as .csv
    -Program came from Aishat UnifiedView_NOMISfeatures_script_function.ipynb 
    within [github]/main/Funtions
    -Program requires:
        1. running the UV cleaning function first for UVclean.csv output
    UnifiedView_Master_script_function
        2. Processsing nomis data into a unified .csv file using function - XXX ??? XXX
        
INPUT:  I_uvNomis_path      stg. file path and name of UV+nomis merged data 
        I_CDRCpath          stg. file path and name of the processed cdrc data 
        I_out_filename      stg. filename only of output csv of UV+nomis+cdrc data 

OUTPUT: O_uv_cdrc          pd of clean UV data 
        [.csv file in current directory of UV_nomis merged data]

RUN: 
            UV_master_path = './UnifiedView_Master.csv'
            nomis_path = 'C:/Users/Neel.Savani-Patel/NEEL_tech/LOCAL_DATA/NOMIS_Features.csv'
            uvnomis_filename = 'UV_nomis_Master.csv'  
            pd_UV_nomis = uv_nomis.uv_merge_nomis(UV_master_path, nomis_path, uvnomis_filename)
         
         
@author:NPS
Created on Tue 2020/01/14
modified: 
Thurs 2020/01/16 -- AO -- L7 to L28
Thurs 2020/01/16 -- AO -- L83 to L91

"""    

def uv_merge_cdrc(I_uvNomis_path, I_CDRCpath, I_out_filename):

    #Load libraries 
    import pandas as pd
    import numpy as np
    
    import numpy as np
    
    import warnings

#    warnings.filterwarnings('ignore') 
#    get_ipython().run_line_magic('matplotlib', 'inline')
#    pd.options.display.max_columns = None
#    pd.options.display.max_rows = None

    #%% B. define inputs      
    uv_path = I_uvNomis_path     # './UV_nomis_Master.csv' - run from previous function
    cdrc_path = I_CDRCpath       # 'Q:/Inform/Strategic Surveillance/Sprint AI/DATA/CDRC_features/CDRC_features.csv'
    o_fname = I_out_filename;


    #%% 1. load in .csv file as a pd both UV_nomis and CDRC
    
    UnifiedView_NOMIS = pd.read_csv(uv_path)
    CDRC_features = pd.read_csv(cdrc_path)
    
    #Delete unwanted column(s)
    UnifiedView_NOMIS = UnifiedView_NOMIS.drop(columns = 'Unnamed: 0')
    #CDRC_features = CDRC_features.drop(columns = 'Unnamed: 0')

 
    #%% 2. prepare NOMIS and merge
    #Drop column(s)
    CDRC_features = CDRC_features.drop(columns = 'Unnamed: 0')


    ##Mergers
    uv_NOMIS_CDRC = pd.merge(UnifiedView_NOMIS, CDRC_features, on = 'Postcode2', how = "inner")
    #Rename column(s)
    uv_NOMIS_CDRC = uv_NOMIS_CDRC.rename(columns = {'Postcode_Sector_x': 'Postcode_Sector'})
    #Drop column(s)
    uv_NOMIS_CDRC = uv_NOMIS_CDRC.drop(['Postcode_Sector_y'], axis = 1)

    
    ### Export Cleaned Unified View Data
    out_fpath = './'
    out_fpath_name = out_fpath + o_fname

    uv_NOMIS_CDRC.to_csv(out_fpath_name)
    ##out_filpath
    #uv_NOMIS_CDRC.to_csv(r"Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/UnifiedView_NOMIS_CDRC.csv")
    ################################################END#################################################
    return uv_NOMIS_CDRC
