#!/usr/bin/env python
# coding: utf-8

# # Unified View & NOMIS Features - Merger

"""
merge_UnifiedView_NOMIS:
    function file to merge nomis data into pre cleaned UV data saved as .csv
    Program came from Aishat UnifiedView_NOMISfeatures_script_function.ipynb within 
    [github]/main/Funtions.
    Program requires:
        1. running the UV cleaning function first for UVclean.csv output
    UnifiedView_Master_script_function
        2. Processsing nomis data into a unified .csv file using function - XXX ??? XXX
        
INPUT:  I_uvclean_path      stg. file path and name of the cleaned UV raw data 
        I_nomispath         stg. file path and name of the processed nomis data 
        I_out_filename      stg. filename only of output csv for clean UV data 

OUTPUT: O_uv_nomis          pd of clean UV data 
        [.csv file in current directory of UV_nomis merged data]

RUN: 
            UV_master_path = './UnifiedView_Master.csv'
            nomis_path = 'C:/Users/Neel.Savani-Patel/NEEL_tech/LOCAL_DATA/NOMIS_Features.csv'
            uvnomis_filename = 'UV_nomis_Master.csv'  
            pd_UV_nomis = uv_nomis.uv_merge_nomis(UV_master_path, nomis_path, uvnomis_filename)
         
         
@author:NPS
Created on Wed 2020/01/14
modified: 
    
"""


def uv_merge_nomis(I_uvclean_path, I_nomispath, I_out_filename):
    
    #Load libraries 
    import pandas as pd
    import numpy as np
    
    import warnings
    
#    warnings.filterwarnings('ignore') 
#    get_ipython().run_line_magic('matplotlib', 'inline')
#    pd.options.display.max_columns = None
#    pd.options.display.max_rows = None


    #%% B. define inputs    
    uv_path = I_uvclean_path          # 'Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/_business_data_fhrs-latest-inspection__view_detail.csv'
    nomis_path = I_nomispath     # 'Q:/Inform/Strategic Surveillance/Sprint AI/DATA/NOMIS_features/NOMIS_Features.csv'
    o_fname = I_out_filename;
    
    #%% 1. load in .csv file as a pd both UV and NOMIS
    
    uni_view = pd.read_csv(uv_path)
    
    nomis_features = pd.read_csv(nomis_path)

    #%% 2. prepare NOMIS and merge
    #Delete unwanted column(s)
    del nomis_features['Unnamed: 0']
    del uni_view['Unnamed: 0']
    
    ##Mergers
    UnifiedView_NOMIS = pd.merge(uni_view, nomis_features, on = 'Postcode_Sector', how = 'inner') 



    #%% 40. return ouputs

    ### Export Cleaned Unified View Data
    #out_filpath = r"Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/"
    out_fpath = './'
    out_fpath_name = out_fpath + o_fname
    
    UnifiedView_NOMIS.to_csv(out_fpath_name)
    # UnifiedView_NOMIS.to_csv(r"Q:/Inform/Strategic Surveillance/Sprint AI/DATA/Unified_View/UnifiedView_NOMIS.csv")
    #############################################END####################################################

    return UnifiedView_NOMIS
