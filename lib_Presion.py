# -*- coding: utf-8 -*-

"""
lib_Presion:
    -Function file to quickly create the necessary steps to do the merging with the master dataset.


INPUT:  I_data_path       stg. file path and name of master data     
        I_out_filename    stg. filename only of output csv for master + Presion

OUTPUT: I_data_Presion            pd of final prepared data with the correct 
                                      feature collection (including Presion)
      [.csv file in current directory of final data filename is I_out_filename]

RUN:    TBD
         
@author: VSA
Created on Tue 2020/02/15
modified: Thr 2020/03/06
    
"""
    

def Presion(I_data_path, I_out_filename, I_data_Presion):  


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
        df = pd.read_csv(I_data_Presion+ i + '.csv')
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

    ## Convertimos la variable Presion_max y Presion_min en numérico
    Provincias_df['Presion_Max'] = Provincias_df['Presion_max'].astype(float)
    Provincias_df['Presion_Min'] = Provincias_df['Presion_min'].astype(float)

    Presion_max_df = Provincias_df[['Año','Mes','Provincia', 'Presion_Max']]
    Presion_max_df_mensual = (Presion_max_df.groupby(('Provincia','Año', 'Mes')) # Agrupar
              .Presion_Max   # Quedarse con la columna IMPORTESALIDA
              .apply(np.mean)       # Calcular su suma
              .reset_index())   # Deshacer índice jerárquico
    Presion_max_df_mensual['Key_merge'] = Presion_max_df_mensual['Provincia'] + Presion_max_df_mensual['Año'].astype(str) + Presion_max_df_mensual['Mes'].astype(str) 

    Presion_min_df = Provincias_df[['Año','Mes','Provincia', 'Presion_Min']]
    Presion_min_df_mensual = (Presion_min_df.groupby(('Provincia','Año', 'Mes')) # Agrupar
              .Presion_Min   # Quedarse con la columna IMPORTESALIDA
              .apply(np.mean)       # Calcular su suma
              .reset_index())   # Deshacer índice jerárquico
    Presion_min_df_mensual['Key_merge'] = Presion_min_df_mensual['Provincia'] + Presion_min_df_mensual['Año'].astype(str) + Presion_min_df_mensual['Mes'].astype(str)


    data['Key_merge'] = data['DESC_PROVINCIA'] + data['Año'].astype(str) + data['Mes'].astype(str)

    Presion_def = pd.merge(Presion_max_df_mensual, Presion_min_df_mensual[['Presion_Min', 'Key_merge']],
                                left_on= 'Key_merge', right_on= 'Key_merge', how='inner')

    ## Merge
    data = pd.merge(data, Presion_def[['Presion_Max', 'Presion_Min', 'Key_merge']],
                                left_on= 'Key_merge', right_on= 'Key_merge', how='left')

    del data['Key_merge']
    
    ### Save as csv:
    out_fpath = './csv/'
    out_fpath_name = out_fpath + o_fname  + '.csv' 
    
    print('Saving dataset to '+str(out_fpath_name))
    data.to_csv(out_fpath_name)

    return Presion_def, data


