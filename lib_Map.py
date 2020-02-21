# -*- coding: utf-8 -*-
import branca.colormap as cm
import pandas as pd
import os
import folium

def plot_map(coord_path, TypeData, Año):

    # Importamos el excel con las coordenadas y hacemos el merge con el conjunto train:
    coordenadas = pd.read_excel(coord_path, sheet_name = "Andalucia")
    Data_map = pd.merge(TypeData, coordenadas[["Provincia", "Población",  "Latitud", "Longitud"]], 
                          left_on= 'DESC_PROVINCIA', right_on= 'Población', how='left')
    Data_map['Latitud'] = Data_map['Latitud'].astype(float)
    Data_map['Longitud'] = Data_map['Longitud'].astype(float)
    
    # Redondeamos la cantidad de salida a numeros enteros
    Data_map['CANTIDADSALIDA_round'] = Data_map['CANTIDADSALIDA'].astype('int')
    
    # Comprobación existe el año dentro del 
    if Año not in Data_map['Año'].unique().tolist():
        del Año
        
    # Filtramos por año (en el caso de que exista)
    if 'Año' in locals():
        año_df = Data_map.loc[Data_map['Año']== Año]
    else:
        año_df = Data_map
        
    # Obtenemos la lista de articulos existentes:
    articulos = año_df['ARTICULO'].unique().tolist()
    
    # For loop para cada articulo
    k= 1
    for k in range(len(articulos)):
        #Filtramos por articulo
        articulos_df = año_df.loc[año_df['ARTICULO']== articulos[k]]
    
        #Columna total del articulo por provincia (Group, hacemos el merge y eliminamos duplicados)
        provincias_suma = articulos_df.groupby('DESC_PROVINCIA', as_index=False)['CANTIDADSALIDA_round'].sum()
        articulos_df2= pd.merge(provincias_suma, articulos_df[['ARTICULO','DESC_PROVINCIA','Latitud','Longitud']],
                            left_on= 'DESC_PROVINCIA', right_on= 'DESC_PROVINCIA', how='left')
        articulos_df3 = articulos_df2.drop_duplicates()
        a=[list(range(0,len(articulos_df3)))]
        articulos_df3=articulos_df3.set_index(a)
    
        # Parámetros:
        Lat_med=articulos_df3['Latitud'].mean()
        Lon_med=articulos_df3['Longitud'].mean()
        max_value = articulos_df3['CANTIDADSALIDA_round'].max() 
        min_value = articulos_df3['CANTIDADSALIDA_round'].min() 
    
        # Tamaño del radio igual a la CANTIDADSALIDA_round normalizada
        if min_value != max_value:
            articulos_df3['Size']= (articulos_df3['CANTIDADSALIDA_round'] - min_value) / (max_value - min_value) + 0.2
        else:
            articulos_df3['Size']= articulos_df3['CANTIDADSALIDA_round']/articulos_df3['CANTIDADSALIDA_round']
    
        # Creación variable: Media de la catidad de salida del articulo partido del numero centros de la provincia
        l1, l2 = [], []
        for i in año_df['DESC_PROVINCIA'].unique().tolist():
            A = año_df.loc[año_df['DESC_PROVINCIA']== i ]
            l1.append(i)
            l2.append(len(A['CENTROCONSUMO'].unique().tolist()))
        df_CANT_CENTROCONSUMO = pd.DataFrame({'DESC_PROVINCIA': l1, 'CANT_CENTROCONSUMO': l2})
        articulos_df4 = pd.merge(articulos_df3, df_CANT_CENTROCONSUMO, on= 'DESC_PROVINCIA', how='left')
        articulos_df4['Media_CANTIDADSALIDA_per_Centro'] = round(articulos_df4['CANTIDADSALIDA_round']/articulos_df4['CANT_CENTROCONSUMO'], 2)
    
        #Creación de la barra de color de la bubble:
        rangeFrac = 0.8
        A=abs(articulos_df4['Media_CANTIDADSALIDA_per_Centro'].min())*rangeFrac
        B=abs(articulos_df4['Media_CANTIDADSALIDA_per_Centro'].max())
        linear = cm.LinearColormap(['white','royalblue'], vmin=A, vmax=B)
    
        # Creación del mapa
        m = folium.Map(location=[Lat_med,Lon_med],zoom_start = 7.2) 
        for i in range(0,len(articulos_df4)):
            #Color dependiendo de Media_CANTIDADSALIDA_per_Centro
            color = linear(articulos_df4.iloc[i]['Media_CANTIDADSALIDA_per_Centro']), 
            #Creación del bubble
            folium.Circle(
                location = [articulos_df4.iloc[i]['Latitud'], articulos_df4.iloc[i]['Longitud']],
                popup = folium.Popup("<b><u>"+articulos_df4.iloc[i]['DESC_PROVINCIA']+'</u>'+
                            ': (' + articulos_df4.iloc[i]['CANT_CENTROCONSUMO'].astype('str')+' centros)'+'</b>'+
                            '<br> Cantidad Salida total: '+ articulos_df4.iloc[i]['CANTIDADSALIDA_round'].astype('str')+
    
                            '<br> Cantidad Salida por Centro: '+ articulos_df4.iloc[i]['Media_CANTIDADSALIDA_per_Centro'].astype('str'),
                            max_width=2650),
                radius= 60000*articulos_df4.iloc[i]['Size'],
                fillColor= linear(articulos_df4.iloc[i]['Media_CANTIDADSALIDA_per_Centro']),
                color=color, fill=True,
            ).add_to(m)
    
        if 'Año' in locals():
            title_html = '''
                     <h3 align="center" style="font-size:20px"><b>'''+ 'Cantidad total articulo ' +str(articulos[k]) +  ': ' +str(articulos_df4['CANTIDADSALIDA_round'].sum()) + ' (Año: '+str(Año)+')'+'''</b></h3>
                     '''
        else:
            title_html = '''
                     <h3 align="center" style="font-size:20px"><b>'''+ 'Cantidad total articulo ' +str(articulos[k]) +  ': ' +str(articulos_df4['CANTIDADSALIDA_round'].sum()) +'''</b></h3>
                     '''
    
        m.get_root().html.add_child(folium.Element(title_html))
        if 'Año' in locals():
            path = './maps/'+ str(Año)
        else:
            path = './maps/Total'
        os.makedirs(path, exist_ok=True)
        m.save(outfile= path + '/'+str(articulos[k]) + ".html")
    