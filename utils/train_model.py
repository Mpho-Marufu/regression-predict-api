"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Imputing missing values and scaling values
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
riders = pd.read_csv('data/riders.csv')
train_data = pd.merge(train, riders, how='inner', on='Rider Id')

train_data['Precipitation in millimeters'].fillna(value=0, inplace=True)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(train_data[["Temperature"]])
train_data["Temperature"]=imputer.transform(train_data[["Temperature"]]).ravel()
#Change time to float
train_data['Placement - Time'] = pd.to_timedelta(train_data['Placement - Time'])
train_data['Placement - Time'] = train_data['Placement - Time'] / pd.offsets.Minute(60)
train_data['Confirmation - Time'] = pd.to_timedelta(train_data['Confirmation - Time'])
train_data['Confirmation - Time'] = train_data['Confirmation - Time'] / pd.offsets.Minute(60)
train_data['Arrival at Destination - Time'] = pd.to_timedelta(train_data['Arrival at Destination - Time'])
train_data['Arrival at Destination - Time'] = train_data['Arrival at Destination - Time'] / pd.offsets.Minute(60)
train_data['Pickup - Time'] = pd.to_timedelta(train_data['Pickup - Time'])
train_data['Pickup - Time'] = train_data['Pickup - Time'] / pd.offsets.Minute(60)
train_data['Arrival at Pickup - Time'] = pd.to_timedelta(train_data['Arrival at Pickup - Time'])
train_data['Arrival at Pickup - Time'] = train_data['Arrival at Pickup - Time'] / pd.offsets.Minute(60)
#Change time to x,y to rightfully reflect a cycle form
train_data['Placement_time_x']=np.sin(2.*np.pi*train_data['Placement - Time']/24.)
train_data['Placement_time_y']=np.cos(2.*np.pi*train_data['Placement - Time']/24.)
train_data['Confirmation - time_x']=np.sin(2.*np.pi*train_data['Confirmation - Time']/24.)
train_data['Confirmation - time_y']=np.cos(2.*np.pi*train_data['Confirmation - Time']/24.)
train_data['Arrival at Destination - time_x']=np.sin(2.*np.pi*train_data['Arrival at Destination - Time']/24.)
train_data['Arrival at Destination - time_y']=np.cos(2.*np.pi*train_data['Arrival at Destination - Time']/24.)
train_data['Pickup - time_x']=np.sin(2.*np.pi*train_data['Pickup - Time']/24.)
train_data['Pickup - time_y']=np.cos(2.*np.pi*train_data['Pickup - Time']/24.)
train_data['Arrival at Pickup - time_x']=np.sin(2.*np.pi*train_data['Arrival at Pickup - Time']/24.)
train_data['Arrival at Pickup - time_y']=np.cos(2.*np.pi*train_data['Arrival at Pickup - Time']/24.)
train_data.drop(['Placement - Time', 'Confirmation - Time', 'Arrival at Destination - Time', 'Pickup - Time', 'Arrival at Pickup - Time'], axis=1, inplace=True)
train_data.drop(['Order No', 'User Id', 'Vehicle Type', 'Rider Id'], axis = 1, inplace = True)

le = LabelEncoder()
train_data['Personal or Business'] = le.fit_transform(train_data['Personal or Business'])

train_data['Platform Type'] = train_data['Platform Type'].replace([1, 2, 3, 4], ['Type_1', 'Type_2', 'Type_3', 'Type_4'])
traindata = pd.get_dummies(train_data, drop_first=True)

traindata.drop('No_of_Ratings', axis=1, inplace=True)
traindata.drop(['Arrival at Pickup - time_x', 'Arrival at Pickup - time_y', 'Arrival at Pickup - Day of Month', 'Arrival at Pickup - Weekday (Mo = 1)'], axis=1, inplace=True)
traindata.drop(['Placement - Day of Month', 'Pickup - Day of Month', 'Arrival at Destination - Day of Month', 'Placement - Weekday (Mo = 1)', 'Pickup - Weekday (Mo = 1)', 'Arrival at Destination - Weekday (Mo = 1)'], axis=1, inplace=True)
column_titles = [col for col in traindata.columns if col!= 'Time from Pickup to Arrival'] + ['Time from Pickup to Arrival']
traindata=traindata.reindex(columns=column_titles)

cols_dropped = ['Arrival at Destination - time_x', 'Arrival at Destination - time_y']
traindata.drop(cols_dropped, axis=1, inplace=True)

X = traindata.iloc[:, :-1].values
y = traindata.iloc[:, -1].values
X_train, X_cross, y_train, y_cross = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit model
regressor = RandomForestRegressor(n_estimators = 75, random_state = 0)
print ("Training Model...")
regressor.fit(X_train, y_train)

# Pickle model for use within our API

save_path = '../assets/trained-models/randomforest_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(regressor, open(save_path,'wb'))
