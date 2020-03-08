# -*- coding: utf-8 -*-

"""
lib_Tiempo:
    -Function file to quickly create the necessary steps to do the merging with the master dataset.


INPUT:  I_data_path       stg. file path and name of master data     
        I_out_filename    stg. filename only of output csv for master + Tiempo

OUTPUT: I_data_Tiempo           pd of final prepared data with the correct 
                                      feature collection (including Tiempo)
      [.csv file in current directory of final data filename is I_out_filename]

RUN:    TBD
         
@author: VSA
Created on Tue 2020/02/22
modified: Thr 2020/03/06
    
"""
    

def tiempo(I_data_path, I_out_filename, I_data_tiempo):  
    
    #Load libraries
    import pandas as pd
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
        df = pd.read_csv(I_data_tiempo + i + '.csv')
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
    Provincias_df=Provincias_df.reset_index(drop=True)

    ## Limpieza para los valores IP dentro de precipitaciones
    l = []
    for i in range(len(Provincias_df['Precipitación: l/m2'])):
        if Provincias_df['Precipitación: l/m2'][i] == 'Ip':
            l.append('0')
        elif Provincias_df['Precipitación: l/m2'][i] =='Acum':
            l.append('0')
        else:
            l.append(Provincias_df['Precipitación: l/m2'][i])
    Provincias_df['Precipitacion'] = l
    #Provincias_df=Provincias_df.dropna()

    ## Eliminamos el sobrante y convertimos la variable precipitaciones en numérico
    del Provincias_df['Precipitación: l/m2']
    Provincias_df['Precipitacion'] = Provincias_df['Precipitacion'].astype(float)
    Provincias_df = Provincias_df.rename(columns={'Horas Sol':'Horas_Sol'})

    Precipitaciones_df = Provincias_df[['Año','Mes','Provincia', 'Precipitacion']]
    Precipitaciones_df_mensual = (Precipitaciones_df.groupby(('Provincia','Año', 'Mes')) # Agrupar
              .Precipitacion   # Quedarse con la columna IMPORTESALIDA
              .apply(sum)       # Calcular su suma
              .reset_index())   # Deshacer índice jerárquico
    Precipitaciones_df_mensual['Key_merge'] = Precipitaciones_df_mensual['Provincia'] + Precipitaciones_df_mensual['Año'].astype(str) + Precipitaciones_df_mensual['Mes'].astype(str) 

    HorasSol_df = Provincias_df[['Año','Mes','Provincia', 'Horas_Sol']]
    HorasSol_df_mensual = (HorasSol_df.groupby(('Provincia','Año', 'Mes')) # Agrupar
              .Horas_Sol   # Quedarse con la columna IMPORTESALIDA
              .apply(sum)       # Calcular su suma
              .reset_index())   # Deshacer índice jerárquico
    HorasSol_df_mensual['Key_merge'] = HorasSol_df_mensual['Provincia'] + HorasSol_df_mensual['Año'].astype(str) + HorasSol_df_mensual['Mes'].astype(str)

    data['Key_merge'] = data['DESC_PROVINCIA'] + data['Año'].astype(str) + data['Mes'].astype(str)

    tiempo_def = pd.merge(Precipitaciones_df_mensual, HorasSol_df_mensual[['Horas_Sol', 'Key_merge']],
                                left_on= 'Key_merge', right_on= 'Key_merge', how='inner')


    ## Merge
    data = pd.merge(data, tiempo_def[['Horas_Sol', 'Precipitacion', 'Key_merge']],
                                left_on= 'Key_merge', right_on= 'Key_merge', how='left')

    del data['Key_merge']
    
    ### Save as csv:
    out_fpath = './csv/'
    out_fpath_name = out_fpath + o_fname  + '.csv' 
    
    print('Saving dataset to '+str(out_fpath_name))
    data.to_csv(out_fpath_name)

    return tiempo_def, data


