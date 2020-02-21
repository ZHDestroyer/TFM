# -*- coding: utf-8 -*-

from lib_UVMasterClean import UnifiedView_cleaning
from lib_mergeUV_NOMIS import uv_merge_nomis
from lib_mergeUV_CDRC import uv_merge_cdrc
import lib_UVDerive as uv_D
from lib_Options import Master_options
from lib_DeleteRubbish import delete_rubbish
from lib_Dummies import Master_dummies
import pandas as pd
from joblib import load
import os


def master_function(rawUVpath, ukpostcode, nomis_path, cdrc_path, I_csvfilename_of_features, test_X_path, std_scaler_path):
    os.makedirs('./csv', exist_ok=True)
    # Load the dataset:
    Data0 = pd.read_csv(rawUVpath)
    print('Input Dataset:')
    display(Data0.head())
    
    # 1. clean the UV raw data [~20 min to run]
    print('1. Cleaning the UV raw data...')
    uv_clean_filename = './csv/UnifiedView_Clean.csv'
    Data1 = UnifiedView_cleaning(rawUVpath, ukpostcode, uv_clean_filename)    
    print('Completed')
    #display(Data1.head())
    # 2. merge with NOMIS    [~10 min to run - memory issue warning]
    print('2. Merging with NOMIS...')
    UV_master_path = './csv/UnifiedView_Clean.csv'
    uvnomis_filename = './csv/UV_nomis.csv'
    Data2 = uv_merge_nomis(UV_master_path, nomis_path, uvnomis_filename)
    print('Completed')
    #display(Data2.head())
    
    # 3. merge with CDRC (~10min to run)
    print('3. Merging with CDRC...')
    uvnomis_path = './csv/UV_nomis.csv'
    uvcdrc_filename = './csv/UV_nomis_cdrc.csv'
    Data3 = uv_merge_cdrc(uvnomis_path, cdrc_path, uvcdrc_filename)
    print('Completed')
    #display(Data3.head())
    
    # Merge with derived data products (~4 min to run)

    #%%  4  SEASONALITY of inspection
    print('4a  SEASONALITY of inspection')
    uvcdrc_path = './csv/UV_nomis_cdrc.csv'  # the output .csv file created in cell above
    uvder_name01 = './csv/UV_derive1.csv' # define file name of output .csv file
    Data4 = uv_D.uv_seasonality(uvcdrc_path, uvder_name01)
    print('Completed')
    #display(Data4.head())
    
    # %%  5  CHAIN RESTURANT (~4 min to run)
    print('4b  CHAIN RESTURANT')
    uv_path = './csv/UV_derive1.csv'
    uvder_name02 = './csv/UV_derive2.csv'
    Data5 = uv_D.uv_chain(uv_path, uvder_name02)
    print('Completed')
    #display(Data5.head())
    
    #5. DEFINE FULL AND TOTAL MASTERLIST OF EVERYTHING
    print('5. DEFINE FULL AND TOTAL MASTERLIST OF EVERYTHING')
    MASTER_FINAL = './csv/uv_COMPLETED.csv'
    Data5.to_csv(MASTER_FINAL)
    print('Completed')
    #display(Data5.head())
    
    # 6. Select the options files [~2-3 min to run]
    print('6. Select the options files')
    # Identfy the file location of the master dataset
    I_data_path = './csv/uv_COMPLETED.csv'
    # Select the keys
    I_keyarray = [1, 1, 1, 1, 1, 1, 1, 1]
    # define file name of output .csv file
    I_out_filename = 'master'
    # function
    keylist, Data6 = Master_options(I_data_path, I_keyarray, I_out_filename)
    print('Completed')
    #display(Data6.head())
    
    # 7. Delete all the features from one csv  [~2-3 min to run]
    # identfy the file location of the master dataset
    print('7. Delete all the features from one csv')
    I_pd_Masterlist = './csv/master_'+ keylist+ '.csv' 
    # define file name of output .csv file
    I_out_filename = 'clean_master_'+ keylist
    # To see the delete list: (Options: 'Yes', 'No')
    Bool = 'No' 
    # function
    Data7 = delete_rubbish(I_pd_Masterlist, I_csvfilename_of_features, I_out_filename, Bool)
    #display(Data7.head())
    
    
    # 8. Create and add dummies features to the master csv (and delete categorical ones)  [~1-2 min to run]
    # identfy the file location of the master dataset
    print('8. Create and add dummies features to the master csv')
    I_data_path = './csv/clean_master_'+ keylist+ '.csv' 
    # define file name of output .csv file
    I_out_filename = 'master_dummies_'+ keylist
    # Max number of dummies for each feature allowed (if ones have less that Max, those features will be removed)
    Max = 50
    # function
    Data8 = Master_dummies(I_data_path, I_out_filename, Max)
    #display(Data8.head())
    
    # 9. Load and add the columns missed
    print('9. Adding some missed dummies...')
    test_X = pd.read_csv(test_X_path)
    del test_X['Unnamed: 0']
    
    if 'summaryScore' in Data8.columns:
        Test_target = Data8.loc[:, 'summaryScore']
        Test_target.to_csv('./csv/final_output_Test_target_'+ keylist+ '.csv')
    Test_features = Data8.loc[:, test_X.columns.tolist()]
    Test_features.fillna(int(0),inplace=True)
    #display(Test_features.head())
    
    # 10. Normalise
    print('10. Normalising...')
    sc=load(std_scaler_path)
    Test_features = sc.transform(Test_features)
    print('Completed')
    # 11. Master dataset
    print('11. Master dataset:')
    Test_features_df = pd.DataFrame(Test_features, columns= test_X.columns.tolist())    
    display(Test_features_df.head())
    
    print('Saving Master dataset to ./csv/final_output_Test_features_'+ keylist+ '.csv')          
    Test_features_df.to_csv('./csv/final_output_Test_features_'+ keylist+ '.csv')
    
    print('Done')
    return Test_features_df
