# -*- coding: utf-8 -*-



"""
NNTRegressor:
    -Function file to quickly create and train neural network

I_data_path, I_Model_path, k, num_epochs, batch_Size, Verbose, NNTList

INPUT:  I_data_path                         stg. file path and name of final merged master data  
        I_Model_path                        stg. file path and name of model outputs
        k                                   integer number which means k in K-cross validation
        num_epochs                          integer number which means the number of epochs 
        batch_Size                          integer number which means the size of the batch in the NNT
        NNTList                             list of int. which meansthe types of Models 
                                            (Possibilities: 1 to 6, you can choose how many you want e.g: [2,3,1,6])



OUTPUT: 
        train_data_df, train_targets_df, test_data_df, test_targets_df
  
RUN:    
         
         
@author: VSA
Created on Thu 2020/02/24
modified:  VSA Mon 2020/03/06
    
"""

def NNTRegressor(I_data_path, I_Model_path, k, num_epochs, batch_Size, Verbose, NNTList):
    
    import numpy as np
    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import time
    import os

    import warnings
    warnings.filterwarnings('ignore') 
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_absolute_error
    
    import keras
    from keras.models import Sequential
    from keras import models
    from keras import layers
    get_ipython().run_line_magic('matplotlib', 'inline')
    from joblib import dump 

    
    #################################### Functions ####################################
    
    # Normalise
    def Normalise(Data, Target):
        sc = StandardScaler()
        dfX = Data.drop([Target], axis=1)
        sc.fit(dfX)
        X = sc.transform(dfX)
        os.makedirs(I_Model_path + 'temp', exist_ok=True)
        print('save scaler to '+ I_Model_path + 'temp/scaler.pickle')
        with open(I_Model_path + 'temp/scaler.pickle', 'wb') as f:
            pickle.dump(sc, f)
        dump(sc, I_Model_path + 'temp/std_scaler.bin', compress=True)
        y = Data.values[:, Data.columns.get_loc(Target)]
        return X, y
    
        
    #Validation
    def validation(train_data, train_targets, num_val_samples, i):
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        return val_data, val_targets

    def partial_train(train_data, train_targets, num_val_samples, i):
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
        axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
        axis=0)
        return partial_train_data, partial_train_targets    

    #Models
    def build_model1(titles):
        red = Sequential()
        red.add(layers.Dense(64, activation='relu', input_shape=(n,)))
        red.add(layers.Dense(128, activation='relu'))
        red.add(layers.Dense(64, activation='relu'))
        red.add(layers.Dense(1))
        red.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        title = 'Three hidden layers (relu: 64, 128, 64) and rmsprop optimizer'
        titles.append(title)
        return red

    def build_model2(titles):
        red = Sequential()
        red.add(layers.Dense(16, activation='relu', input_shape=(n,)))
        red.add(layers.Dense(16, activation='relu'))
        red.add(layers.Dense(1))
        red.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        title = 'Two hidden layers (relu: 16) and rmsprop optimizer'
        titles.append(title)
        return red

    def build_model3(titles):    
        red = Sequential()
        red.add(layers.Dense(512, activation='relu', input_shape=(n,)))
        red.add(layers.Dense(1))
        red.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        title = 'One hidden layers (relu: 512) and rmsprop optimizer'
        titles.append(title)
        return red

    def build_model4(title):    
        red = Sequential()
        red.add(layers.Dense(64, activation='relu', input_shape=(n,)))
        red.add(layers.Dense(64, activation='relu'))   
        red.add(layers.Dense(64, activation='softmax'))
        red.add(layers.Dense(1))
        red.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        title = 'Three hidden layers (relu: 64) and rmsprop optimizer'
        titles.append(title)
        return red

    def build_model5(title):    
        red = Sequential()
        red.add(layers.Dense(32, activation='relu', input_shape=(n,)))
        red.add(layers.Dense(64, activation='relu'))   
        red.add(layers.Dense(128, activation='relu'))
        red.add(layers.Dense(128, activation='relu'))
        red.add(layers.Dense(512, activation='relu'))
        red.add(layers.Dense(1))
        red.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        title = 'One hidden layers (sigmoid: 16) and rmsprop optimizer'
        titles.append(title)
        return red
  
    def build_model6(titles):
        red = Sequential()
        red.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
        red.add(layers.Dense(64, activation='relu'))
        red.add(layers.Dense(64, activation='relu'))
        red.add(layers.Dense(64, activation='relu'))
        red.add(layers.Dense(1))
        red.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        title = 'Four hidden layers (relu: 64) and rmsprop optimizer'
        titles.append(title)
        return red  
    

    
    #MAE in k
    def average_mae(train_all_scores, val_all_scores, num_epochs):
        training = [np.mean([x[k] for x in train_all_scores]) for k in range(num_epochs)]
        validation = [np.mean([x[k] for x in val_all_scores]) for k in range(num_epochs)]
        return training, validation
    
    #Visuals
    def plot_Mae(average_mae_training, average_mae_validation, title, i):
        my_dpi=96
        plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
        plt.plot(range(1, len(average_mae_training) + 1), average_mae_training, 'o', label='Training')
        plt.plot(range(1, len(average_mae_validation) + 1), average_mae_validation, 'lightseagreen', label='Validation')
        plt.title(title, size=18)
        plt.xlabel('Epochs', size=18)
        plt.ylabel('Mean Absolute Error', size=18)
        plt.legend(loc='best',prop={'size': 16}) 
        os.makedirs(I_Model_path + 'visuals', exist_ok=True)
        plt.savefig(I_Model_path + 'visuals/MAE Model' + str(NNTList[i]) + '.png') 
        
        
    # Timer
    def Timer(Model_Step, last_time, k):
        Time_list = []
        now = time.time() - last_time
        if now < 60:
            print('   Time of '+ Model_Step+str(k)+': '+ str(round(now))+'s')
            Time_list.append('Time of '+ Model_Step +str(k)+': '+ str(round(now))+'s')
        elif 60<now<3600:
            now_m = now/60
            minutes = int(now_m)
            secs = int((now_m - minutes)*60)
            print('   Time of '+ Model_Step+str(k)+': ' +str(minutes)+'m and '+str(secs)+'s')
            Time_list.append('   Time of '+ Model_Step+str(k)+': ' +str(minutes)+'m and '+str(secs)+'s')
            del now_m
            del minutes
            del secs
        elif now>3600:
            now_h = now/3600
            hours = int(now_h)
            now_m = (now_h - hours)*60
            minutes = int(now_m)
            secs = int((now_m - minutes)*60)
            print('   Time of '+ Model_Step+str(k)+': ' +str(hours)+'h, '+str(minutes)+'m and '+str(secs)+'s')
            Time_list.append('   Time of '+ Model_Step+str(k)+': ' +str(hours)+'h, '+str(minutes)+'m and '+str(secs)+'s')
            del now_h
            del hours
            del now_m
            del minutes
            del secs
            
    ############################################################################################################
    
    # Define inputs
    Master_path = I_data_path    
    os.makedirs(I_Model_path, exist_ok=True)
    
    ## Load data:
    print('Loading the data...')
    data = pd.read_csv(Master_path)

    
    #Creating a dataset copy
    Data = data.copy()
    print('Working dataset has: ' +  str(Data.shape[0])+ ' rows and ' + str(Data.shape[1])+ ' columns')
    

    # Normalise
    print('Normalising...')
    X, y = Normalise(Data,'CANTIDADSALIDA')
    print('Completed')
    
    #Split in train test
    print('Spliting in train and test set')
    train_data, test_data, train_targets, test_targets = train_test_split(X, y, 
                                                                          train_size=0.80, test_size=0.20, random_state=0)

    
    
    ### Save as csv:
    os.makedirs(I_Model_path + 'temp', exist_ok=True)
    dfX = Data.drop('CANTIDADSALIDA', axis=1)
    
    print('Saving training feature set to ' + I_Model_path + 'temp/train_X.csv')
    train_data_df = pd.DataFrame(train_data, columns= dfX.columns )
    train_data_df.to_csv(I_Model_path + 'temp/train_X.csv')
    with open(I_Model_path + 'temp/train_X.pickle', 'wb') as f:
        pickle.dump(train_data, f)    
    
    print('Saving training labels to '+I_Model_path + 'temp/train_Y.csv')
    train_targets_df = pd.DataFrame(train_targets, columns= ['CANTIDADSALIDA'])
    train_targets_df.to_csv(I_Model_path + 'temp/train_Y.csv')
    with open(I_Model_path + 'temp/train_Y.pickle', 'wb') as f:
        pickle.dump(train_targets, f) 

    print('Saving test feature set to '+ I_Model_path + 'temp/test_X.csv')
    test_data_df = pd.DataFrame(test_data, columns= dfX.columns )
    test_data_df.to_csv(I_Model_path + 'temp/test_X.csv')
    with open(I_Model_path + 'temp/test_X.pickle', 'wb') as f:
        pickle.dump(test_data, f)    
    
    print('Saving test labels to '+I_Model_path + '/temp/test_Y.csv')
    test_targets_df = pd.DataFrame(test_targets, columns= ['CANTIDADSALIDA'])
    test_targets_df.to_csv(I_Model_path + 'temp/test_Y.csv')
    with open(I_Model_path + 'temp/test_Y.pickle', 'wb') as f:
        pickle.dump(test_targets, f)
        

    
    train_data_df["CANTIDADSALIDA"] = train_targets_df["CANTIDADSALIDA"]
    train_data_df['CANTIDADSALIDA'] = train_data_df['CANTIDADSALIDA'].astype(int)
    
    test_data_df["CANTIDADSALIDA"] = test_targets_df["CANTIDADSALIDA"]
    test_data_df['CANTIDADSALIDA'] = test_data_df['CANTIDADSALIDA'].astype(int)
    
        
    # Parameters
    #A) Lists:
    models = []
    titles = [] 
    
    #B) Temps
    n = train_data.shape[1] # number of features
    num_val_samples = len(train_data) // k

    #Model
    if 1 in NNTList:
        models.append(build_model1(titles)) 
    if 2 in NNTList:
        models.append(build_model2(titles)) 
    if 3 in NNTList:
        models.append(build_model3(titles)) 
    if 4 in NNTList:
        models.append(build_model4(titles)) 
    if 5 in NNTList:
        models.append(build_model5(titles)) 
    if 6 in NNTList:
        models.append(build_model6(titles)) 
              
    # Training K-Cross Neural Network:
    print('- Starting to train the model:')
    final_time = time.time()
    for i in range(len(models)):
        last_time = time.time()

        print('Model '+ str(NNTList[i]) + ':')
        # NNT model 
        red = models[i]

        # Lists    
        train_all_scores_byModel = []
        val_all_scores_byModel = []

        #K-Cross-validation
        for j in range(k):

            last_time2 = time.time()
            print("   Processing fold #", j)
            
            # Validation
            val_data, val_targets = validation(train_data, train_targets, num_val_samples, j)
            partial_train_data, partial_train_targets = partial_train(train_data, train_targets, num_val_samples, j)

            # Training
            training = red.fit(partial_train_data, partial_train_targets, validation_data = (val_data, val_targets),
                                epochs = num_epochs, batch_size = batch_Size, verbose=Verbose)

            #Mean absolute error
            train_mae_training = training.history['mae']
            val_mae_training = training.history['val_mae']

            # Adding to the list
            train_all_scores_byModel.append(train_mae_training)
            val_all_scores_byModel.append(val_mae_training)

            #timer
            Timer('Step ', last_time2, j)

            del val_data, val_targets

        #Saving the model 
        print('Saving temps')  
        
        os.makedirs(I_Model_path + 'models', exist_ok=True)
        os.makedirs(I_Model_path + 'temp', exist_ok=True)
        
        red.save(I_Model_path + 'models/Model' +str(NNTList[i])+ '.h5')  
            
        with open(I_Model_path + 'models/NNTModel' +str(NNTList[i])+ '.pickle', 'wb') as f:
            pickle.dump(red, f)
        #timer
        print('   ' + str(Timer('Model ', last_time, NNTList[i])))

        # Evaluate
        test_mse_score, test_mae_score = red.evaluate(test_data, test_targets,verbose=0)
        print("      Mean Absolute Error model "+ str(NNTList[i])+ ":"  "= {:.2f}".format(test_mae_score))
        average_mae_training, average_mae_validation =  average_mae(train_all_scores_byModel, val_all_scores_byModel, num_epochs)

        with open(I_Model_path + 'temp/NNT_error' +str(NNTList[i])+ '.pickle', 'wb') as f:
            pickle.dump(test_mae_score, f)        
        
        # Visual 
        plot_Mae(average_mae_training, average_mae_validation, titles[i], i)  
        plt.rcdefaults()
    print('      ' + str(Timer('Global', final_time, 0)))   
    



    return  train_data_df, train_targets_df, test_data_df, test_targets_df



