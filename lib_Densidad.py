# -*- coding: utf-8 -*-

"""
lib_Densidad:
    -Function file to quickly create the necessary steps to do the merging with the master dataset.


INPUT:  I_data_path       stg. file path and name of master data     
        I_out_filename    stg. filename only of output csv for master + densidad

OUTPUT: I_data_densidad             pd of final prepared data with the correct 
                                      feature collection (including densidad)
      [.csv file in current directory of final data filename is I_out_filename]

RUN:    TBD
         
@author: VSA
Created on Tue 2020/02/14
modified: Thr 2020/03/06
    
"""
    

def densidad(I_data_path, I_out_filename, I_data_densidad):  
    
    #Load libraries
    import pandas as pd
    import numpy as np
    import warnings
    from datetime import datetime
    warnings.filterwarnings('ignore') 
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    # Define inputs
    Master_path = I_data_path     
    o_fname = I_out_filename;

    ## Load data:
    data = pd.read_csv(Master_path)
    del data['Unnamed: 0']  
    
    ## Provincias
    provincias_list = ['Almería', 'Cádiz', 'Córdoba', 'Granada', 'Huelva', 'Jaén', 'Málaga', 'Sevilla']

    ## Cargando datos 
    provincias_list_csv = []
    Provincias_df = pd.DataFrame()
    for i in provincias_list:
        df = pd.read_excel(I_data_densidad + i + '.xlsx', sheet_name='tabla-9687')
        df['Key_merge'] = df['Fecha']+i
        Provincias_df = Provincias_df.append(df, ignore_index = True)


    # Densidad: 0-4 años
    Provincias_df['Densidad: 0-4 años']=Provincias_df.iloc[:, 2] 
    for i in range(3, 7):    
        Provincias_df['Densidad: 0-4 años']+=Provincias_df.iloc[:, i] 

    # Densidad: 5-18 años
    Provincias_df['Densidad: 5-18 años']=Provincias_df.iloc[:, 7] 
    for i in range(8, 21):    
        Provincias_df['Densidad: 5-18 años']+=Provincias_df.iloc[:, i] 

    # Densidad: 19-30 años
    Provincias_df['Densidad: 19-30 años']=Provincias_df.iloc[:, 21] 
    for i in range(22, 33):    
        Provincias_df['Densidad: 19-30 años']+=Provincias_df.iloc[:, i] 

    # Densidad: 31-45 años
    Provincias_df['Densidad: 31-45 años']=Provincias_df.iloc[:, 33] 
    for i in range(34, 48):    
        Provincias_df['Densidad: 31-45 años']+=Provincias_df.iloc[:, i] 

    # Densidad: 46-65 años
    Provincias_df['Densidad: 46-65 años']=Provincias_df.iloc[:, 48] 
    for i in range(49, 68):    
        Provincias_df['Densidad: 46-65 años']+=Provincias_df.iloc[:, i] 

    # Densidad: 66-75 años
    Provincias_df['Densidad: 66-75 años']=Provincias_df.iloc[:, 68] 
    for i in range(69, 78):    
        Provincias_df['Densidad: 66-75 años']+=Provincias_df.iloc[:, i] 

    # Densidad: 76> años
    Provincias_df['Densidad: 76> años']=Provincias_df.iloc[:, 78] 
    for i in range(79, 103):    
        Provincias_df['Densidad: 76> años']+=Provincias_df.iloc[:, i] 

     # Creamos la key:
    l_fecha = []
    for i in range(len(data)):
        if data['Mes'][i]<7:
            l_fecha.append('1 de enero de ' + str(data['Año'][i])+data['DESC_PROVINCIA'][i])
        elif data['Mes'][i]>6:
            l_fecha.append('1 de julio de ' + str(data['Año'][i])+data['DESC_PROVINCIA'][i])
    data['Key_merge'] = l_fecha

    ## Merge
    col_merge = ['Key_merge', 'Densidad: 0-4 años', 'Densidad: 5-18 años', 'Densidad: 19-30 años', 'Densidad: 31-45 años', 
                 'Densidad: 46-65 años', 'Densidad: 66-75 años', 'Densidad: 76> años']


    data = pd.merge(data, Provincias_df[col_merge], left_on= 'Key_merge', right_on= 'Key_merge', how='left')

    del data['Key_merge']

    ### Save as csv:
    out_fpath = './csv/'
    out_fpath_name = out_fpath + o_fname  + '.csv' 

    print('Saving dataset to '+str(out_fpath_name))

    return Provincias_df, data


