# -*- coding: utf-8 -*-

"""
Dummies_Preparing_dataset_engineer:
    -Function file to quickly create the necessary type of features needed to run the Neural Network


INPUT:  I_data_path       stg. file path and name of UV master data     
        I_out_filename    stg. filename only of output csv for UV master data + dummies

OUTPUT: O_master_data_key             pd of final prepared data with the correct 
                                      feature collection (including dummies)
      [.csv file in current directory of final data filename is I_out_filename]

Example: 
I_data_path = 'C:/Users/Victor.SalasAranda/Desktop/FSA/csv/options/Master_1_1_1_1_1_1_1_1.csv'
I_out_filename = 'master_dummies'
Max = 50     # Max = float('inf') to delete this bound
pd_Master_dummies = Master_dummies(I_data_path, I_out_filename, Max)

RUN:    TBD
         
@author: VSA
Created on Tue 2020/02/22
modified: Thr 2020/03/06
    
"""
    

def Master_dummies(I_data_path, I_out_filename, Max):  
    
    #Load libraries
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore') 
    
    # Define inputs
    Master_path = I_data_path     
    o_fname = I_out_filename;
    
    
    ## Load data:
    data = pd.read_csv(Master_path)
    del data['Unnamed: 0']  
    Data = data.copy()
    Data['ARTICULO'] = Data['ARTICULO'].astype(str)
    # Data['Año'] = Data['Año'].astype(str)
    Data['Mes'] = Data['Mes'].astype(str) 
    l = ['CENTROCONSUMO', 'ID_ORGANOGESTOR', 'UBICACIONVIRTUAL', 'CENTROCONSUMO_str']
    for i in l:
        if i in Data.columns:
            Data[i] = Data[i].astype(str)

    
    
    ## Dummies:
    categorie_columns = Data.select_dtypes(include=['category']).columns.tolist()
    O_columns = Data.select_dtypes(include=['O']).columns.tolist()
    bool_columns = Data.select_dtypes(include=['bool']).columns.tolist()
    dummies_columns = categorie_columns + O_columns + bool_columns
    

    print('Dummies features: ')
    for i in dummies_columns:
        a = len(Data[i].unique().tolist())
        print('  - ' + i+': ' + str(a) +' new columns')
        if a > Max:
            print('    --> '+ i + ' has been removed')
            del Data[i]
            dummies_columns.remove(i)
            
    listDummies = []
    if dummies_columns != []:
        for i in dummies_columns:
            listDummies.append(pd.get_dummies(Data[i], prefix = i))
            del Data[i]    
            

    print('Adding dummies to dataset...')        
    for Dummy in listDummies:
        for i in Dummy.columns:
            Data[i]=Dummy[i]
    print('Completed')
    #print('Final dataset:')
    #display(Data.head())
    print('Working dataset has: ' +  str(Data.shape[0])+ ' rows and ' + str(Data.shape[1])+ ' columns')
    
    ### Save as csv:
    out_fpath = './csv/'
    out_fpath_name = out_fpath + o_fname  + '.csv' 
    
    print('Saving dataset to '+str(out_fpath_name))
    Data.to_csv(out_fpath_name)

    
    return Data


