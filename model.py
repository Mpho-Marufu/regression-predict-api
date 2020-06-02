"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    # Imputing missing values and scaling values
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder

    test_data = feature_vector_df.copy()
    test_data['Precipitation in millimeters'].fillna(value=0, inplace=True)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(test_data[["Temperature"]])
    test_data["Temperature"]=imputer.transform(test_data[["Temperature"]]).ravel()

    #Change time to float
    test_data['Arrival at Pickup - Time'] = pd.to_timedelta(test_data['Arrival at Pickup - Time'])
    test_data['Arrival at Pickup - Time'] = test_data['Arrival at Pickup - Time'] / pd.offsets.Minute(60)
    test_data['Placement - Time'] = pd.to_timedelta(test_data['Placement - Time'])
    test_data['Placement - Time'] = test_data['Placement - Time'] / pd.offsets.Minute(60)
    test_data['Confirmation - Time'] = pd.to_timedelta(test_data['Confirmation - Time'])
    test_data['Confirmation - Time'] = test_data['Confirmation - Time'] / pd.offsets.Minute(60)
    test_data['Pickup - Time'] = pd.to_timedelta(test_data['Pickup - Time'])
    test_data['Pickup - Time'] = test_data['Pickup - Time'] / pd.offsets.Minute(60)

    #Change time to x,y to rightfully reflect a cycle form
    test_data['Arrival at Pickup - time_x']=np.sin(2.*np.pi*test_data['Arrival at Pickup - Time']/24.)
    test_data['Arrival at Pickup - time_y']=np.cos(2.*np.pi*test_data['Arrival at Pickup - Time']/24.)
    test_data['Placement_time_x']=np.sin(2.*np.pi*test_data['Placement - Time']/24.)
    test_data['Placement_time_y']=np.cos(2.*np.pi*test_data['Placement - Time']/24.)
    test_data['Confirmation - time_x']=np.sin(2.*np.pi*test_data['Confirmation - Time']/24.)
    test_data['Confirmation - time_y']=np.cos(2.*np.pi*test_data['Confirmation - Time']/24.)
    test_data['Pickup - time_x']=np.sin(2.*np.pi*test_data['Pickup - Time']/24.)
    test_data['Pickup - time_y']=np.cos(2.*np.pi*test_data['Pickup - Time']/24.)

    test_data.drop(['Arrival at Pickup - Time', 'Placement - Time', 'Confirmation - Time', 'Pickup - Time'], axis=1, inplace=True)
    test_data.drop(['Order No', 'User Id', 'Vehicle Type', 'Rider Id'], axis = 1, inplace = True)
    
    le = LabelEncoder()
    test_data['Personal or Business'] = le.fit_transform(test_data['Personal or Business'])

    test_data['Platform Type'] = test_data['Platform Type'].replace([1, 2, 3, 4], ['Type_1', 'Type_2', 'Type_3', 'Type_4'])
    testdata = pd.get_dummies(test_data, drop_first=True)

    testdata.drop('No_of_Ratings', axis=1, inplace=True)
    testdata.drop(['Arrival at Pickup - time_x', 'Arrival at Pickup - time_y', 'Arrival at Pickup - Day of Month', 'Arrival at Pickup - Weekday (Mo = 1)'], axis=1, inplace=True)
    testdata.drop(['Placement - Day of Month', 'Pickup - Day of Month', 'Placement - Weekday (Mo = 1)', 'Pickup - Weekday (Mo = 1)'], axis=1, inplace=True)

    # ------------------------------------------------------------------------

    return testdata

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
