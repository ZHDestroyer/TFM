# -*- coding: utf-8 -*-

"""
lib_Temperatura:
    -Function file to quickly create the necessary steps to do the merging with the master dataset.


INPUT:  I_data_path       stg. file path and name of master data     
        I_out_filename    stg. filename only of output csv for master + Temperatura

OUTPUT: I_data_Temperatura            pd of final prepared data with the correct 
                                      feature collection (including Temperatura)
      [.csv file in current directory of final data filename is I_out_filename]

RUN:    TBD
         
@author: VSA
Created on Tue 2020/02/16
modified: Thr 2020/03/06
    
"""
    
    

def temperatura(I_data_path, I_out_filename, I_data_temp):  
    
 
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
    for i in provincias_list:
        df = pd.read_csv(I_data_temp + i + '.csv')
        provincias_list_csv.append(df)

    ## Unión de dataframes
    Provincias_df = pd.DataFrame()
    for i in range(len(provincias_list_csv)):
        l_fecha = []
        for j in provincias_list_csv[i]['FECHA']:
            if "/" in j:
                l_fecha.append(datetime.strptime(j, '%d/%m/%Y'))
            elif "-":
                l_fecha.append(datetime.strptime(j, '%d-%m-%Y'))
        provincias_list_csv[i]['Año'] = [k.year for k in l_fecha]
        provincias_list_csv[i]['Mes'] = [k.month for k in l_fecha]
        provincias_list_csv[i]['Día'] = [k.day for k in l_fecha]
        provincias_list_csv[i]['Provincia'] = [provincias_list[i] for k in provincias_list_csv[i]['FECHA']] 
        Provincias_df = Provincias_df.append(provincias_list_csv[i], ignore_index = True)
    del Provincias_df['FECHA']
    Provincias_df.drop_duplicates(keep='first', inplace=True)
    Provincias_df = Provincias_df.reset_index(drop=True)

    ## Convertimos la variable Temp_Max y Temp_Min en numérico
    Provincias_df['Temp_Max'] = Provincias_df['Temp_Max'].astype(float)
    Provincias_df['Temp_Min'] = Provincias_df['Temp_Min'].astype(float)

    Temperatura_max_df = Provincias_df[['Año','Mes','Provincia', 'Temp_Max']]
    Temperatura_max_df_mensual = (Temperatura_max_df.groupby(('Provincia','Año', 'Mes')) # Agrupar
              .Temp_Max   # Quedarse con la columna IMPORTESALIDA
              .apply(np.mean)       # Calcular su suma
              .reset_index())   # Deshacer índice jerárquico
    Temperatura_max_df_mensual['Key_merge'] = Temperatura_max_df_mensual['Provincia'] + Temperatura_max_df_mensual['Año'].astype(str) + Temperatura_max_df_mensual['Mes'].astype(str) 

    Temperatura_min_df = Provincias_df[['Año','Mes','Provincia', 'Temp_Min']]
    Temperatura_min_df_mensual = (Temperatura_min_df.groupby(('Provincia','Año', 'Mes')) # Agrupar
              .Temp_Min   # Quedarse con la columna IMPORTESALIDA
              .apply(np.mean)       # Calcular su suma
              .reset_index())   # Deshacer índice jerárquico
    Temperatura_min_df_mensual['Key_merge'] = Temperatura_min_df_mensual['Provincia'] + Temperatura_min_df_mensual['Año'].astype(str) + Temperatura_min_df_mensual['Mes'].astype(str)

    data['Key_merge'] = data['DESC_PROVINCIA'] + data['Año'].astype(str) + data['Mes'].astype(str)

    Temperatura_def = pd.merge(Temperatura_max_df_mensual, Temperatura_min_df_mensual[['Temp_Min', 'Key_merge']],
                                left_on= 'Key_merge', right_on= 'Key_merge', how='inner')

    ## Merge
    data = pd.merge(data, Temperatura_def[['Temp_Max', 'Temp_Min', 'Key_merge']],
                                left_on= 'Key_merge', right_on= 'Key_merge', how='left')

    del data['Key_merge']
    
    ### Save as csv:
    out_fpath = './csv/'
    out_fpath_name = out_fpath + o_fname  + '.csv' 
    
    print('Saving dataset to '+str(out_fpath_name))
    data.to_csv(out_fpath_name)

    return Temperatura_def, data


