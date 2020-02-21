#!/usr/bin/env python
# coding: utf-8

def train_RIO_model(encoded_train_X_file, encoded_test_X_file, train_Y_file, test_Y_file, rfr_error_file, rfr_file):
    ''' Input features:
    - encoded_train_X_file : [pickle] {DataFrame} encoded training feature set file e.g. 'temp/encoded_train_X_df.pickle'
    - encoded_test_X_file  : [pickle] {DataFrame} encoded test training set file e.g. 'temp/encoded_test_X_df.pickle'
    - train_Y_file         : [pickle] {DataFrame} training labels file e.g. 'temp/train_Y_df.pickle'
    - test_Y_file          : [pickle] {DataFrame} testing labels file e.g. 'models/test_Y_df.pickle'
    - rfr_error_file       : [pickle] mean absolute error of regression model e.g. 'models/rfr_error.pickle'
    - rfr_file             : [pickle] saved regression model e.g. 'models/rfr.pickle'
    '''
    import pandas as pd
    from rio_sdk.rio_service import RioService
    import pickle
    from sklearn.metrics import mean_absolute_error
    RIO_HOST = "localhost"
    RIO_PORT = 50051
    rio_service = RioService(RIO_HOST, RIO_PORT)
    
    print('Loading encoded training data')
   
    with open(encoded_train_X_file, 'rb') as f:
        encoded_train_X_df = pickle.load(f)
    
    print('Loading encoded test data')
    with open(encoded_test_X_file, 'rb') as f:
        encoded_test_X_df = pickle.load(f)
    
    print('Loading training labels')
    with open(train_Y_file, 'rb') as f:
        train_Y_df = pickle.load(f)
    
    print('Loading test labels')
    with open(test_Y_file, 'rb') as f:
        test_Y_df = pickle.load(f)
        
    with open(rfr_error_file, 'rb') as f:
        error = pickle.load(f)
    # Load trained RandomForestRegressor model
    with open(rfr_file, 'rb') as f:
        rfr = pickle.load(f)
        
    encoded_train_X_df = encoded_train_X_df.astype('float32')
    train_preds = rfr.predict(encoded_train_X_df)
    train_preds_df = pd.DataFrame(train_preds)
    rfr_rio = rio_service.train(train_preds_df, encoded_train_X_df, train_Y_df[['summaryScore']])
    
    def compute_rio_improvement(error, rio_error):
        imp = error - rio_error
        percentage = imp * 100 / error
        return percentage

    # Our model's predictions on encoded_test_X_df are:
    preds = rfr.predict(encoded_test_X_df)
    preds_df = pd.DataFrame(preds)
    # Ask RIO to improve these predictions on encoded_test_X_df
    rio_means, rio_variances = rio_service.predict(encoded_test_X_df, preds_df, rfr_rio)

    error_rio = mean_absolute_error(test_Y_df, rio_means)
    rio_improvement_percent = compute_rio_improvement(error, error_rio)
    print(f"Rio improvement is {rio_improvement_percent:.2f}%")
    print('Rio mean error is', error_rio)
    rio_service.save(rfr_rio, 'models/rfr_rio.bytes')


def RIO_predictor(input_csv, rfr_file, encoded_train_X_file):
    '''Input parameters:
    - input_csv    : [CSV] file with FBO's to predict. Each on seperate row
    - rfr_file     : [pickle] saved regression model e.g. 'models/rfr.pickle' 
    - columns_drop : [dict] of columns to drop from dataset e.g. {'hygieneScore', 'structuralScore'}
    - encoded_train_X_file : [pickle] {DataFrame} encoded training feature set file e.g. 'temp/encoded_train_X_df.pickle'
    Note make sure columns_drop is the same as trained on
    
    '''
    import pandas as pd
    from rio_sdk.rio_service import RioService
    import pickle
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

    RIO_HOST = "localhost"
    RIO_PORT = 50051
    rio_service = RioService(RIO_HOST, RIO_PORT)
    
    # Load model
    with open(rfr_file, 'rb') as f:
        rfr_loaded = pickle.load(f)
        
    # Load Data Sets
    with open(input_csv) as df_file:
        DF = pd.read_csv(input_csv, keep_default_na=False)
    
    # Categorize columns
    cat_columns = DF.select_dtypes(['category']).columns
    DF[cat_columns] = DF[cat_columns].apply(lambda x: x.cat.codes)
    
    # split
    X_df = DF.drop(['summaryScore'], axis=1)
    Y_df = DF[['summaryScore']]
    
    # Predict sinlge FBO for now
    FBO_to_predict_dict = X_df.iloc[0].to_dict()
    
    # Encode Dataset
    def encode_dataset(reference_df, to_encode_df):
        """
        Encodes the passed dataset and makes it contains the same columns, in the same order, as the reference one.
        See https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data
        """
        encoded_df = pd.get_dummies(to_encode_df, sparse=True)
        # Get missing columns in the encoded dataset
        missing_cols = set(reference_df.columns ) - set(encoded_df.columns )
        # Add missing columns in encoded set with default value equal to 0
        for c in missing_cols:
            encoded_df[c] = 0
        # Ensure columns in the encoded set are in the same order as in the reference set
        encoded_df = encoded_df[reference_df.columns]
        return encoded_df
    
    # This dataframes MUST contain all the possible values
    # It becomes the "reference" in terms of encoded columns

    with open(encoded_train_X_file, 'rb') as f:
        encoded_train_X_df = pickle.load(f)
    
    print('Encode set')
    # Convert the dictionary to a DataFrame
    FBO_to_predict_df = pd.DataFrame([FBO_to_predict_dict], columns=FBO_to_predict_dict.keys())
    # Encode the DataFrame
    encoded_FBO_to_predict_df = encode_dataset(encoded_train_X_df, FBO_to_predict_df)
    
    # Make the prediction
    FBO_FHRS_prediction = rfr_loaded.predict(encoded_FBO_to_predict_df)
    print(f"summaryScore prediction: {FBO_FHRS_prediction[0]}")
    
    # Load the RIO model
    rfr_rio_loaded = rio_service.load('models/rfr_rio.bytes')
    
    # Make the RIO prediction
    FBO_FHRS_prediction_df = pd.DataFrame(FBO_FHRS_prediction)
    prediction_mean, prediction_variance = rio_service.predict(encoded_FBO_to_predict_df, FBO_FHRS_prediction_df, rfr_rio_loaded)
    print(f"RIO corrected summarScore prediction: {prediction_mean[0]}, with variance {prediction_variance[0]}")
    
    # 95% confidence interval
    from math import sqrt
    predict_response_mean = prediction_mean[0]
    predict_response_var = prediction_variance[0]
    predict_response_std = sqrt(predict_response_var)
    lower_bound = predict_response_mean - 1.96 * predict_response_std
    upper_bound = predict_response_mean + 1.96 * predict_response_std
    
    print(f"Lower bound: {lower_bound}, mean: {predict_response_mean}, upper bound: {upper_bound}")
    
    from scipy.stats import norm
    # Generate random variates from this distribution
    data_normal = norm.rvs(size=10000,loc=predict_response_mean, scale=predict_response_std)
    
    plt.boxplot(data_normal)
    # plt.boxplot(data_normal, conf_intervals=[[lower_bound, upper_bound]], notch=True)
    plt.xlabel("FBO #")
    plt.ylabel("Predicted summaryScore");
    plt.show()
    
    plt.errorbar(predict_response_mean, 1, xerr=2*predict_response_std, fmt='o')
    plt.xlabel("summaryScore 95% CI")
    plt.yticks((0,1,2), ("","FBO #1",''));
    plt.show()






