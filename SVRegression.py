##### Desenvolvido por Vinícius Roberto Simões Nazato Nº USP 11282009###########
##### Developed by Vinícius Roberto Simões Nazato Nº USP 11282009###########

import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class SVRegression:

    def __init__(self, file_path: str, timesteps= 5):
        self._file = file_path
        self._timesteps = timesteps
        self._df = self.read_data()
        pass

    def read_data(self):

        df = pd.read_excel(self._file, index_col=0)
        df.drop('date',axis=1, inplace=True)

        df.iloc[:,0] = df.shift(-1)
        return df

    def train_test_split(self, show_data = False, train_pct = .7):
        
        #Splits data into training and testing
        train, test = np.split(self._df, [int(train_pct *len(self._df))])
        
        print(train, test)

        if show_data == True:
        
            #Create plot to show Train - Test Division
            _,ax = plt.subplots()
            train.plot(title='Train', ax = ax, )
            test.plot(title='Test', color='orange',  ax = ax)
            ax.legend(['Train', 'Test'])
            plt.title('Train-Test Split')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.show()

        #Scale data to be between [0:1]
        self.scaler = MinMaxScaler()
        train.iloc[:,0] = self.scaler.fit_transform(train)
        test.iloc[:,0] = self.scaler.fit_transform(test)
        
        #transform data into numpy arrays
        train_data = train.values
        test_data = test.values

        #Converting the Numpy arrays in 2D Tensor - [Batch, Timestep]

        train_data_timesteps=np.array([[j for j in train_data[i:i+self._timesteps]] for i in range(0,len(train_data) - self._timesteps + 1)])[:,:,0]
        test_data_timesteps=np.array([[j for j in test_data[i:i+self._timesteps]] for i in range(0,len(test_data) - self._timesteps + 1)])[:,:,0]

        #Divide data into x_train, y_train, x_test, y_test
        self.x_train, self.y_train = train_data_timesteps[:,:self._timesteps - 1], train_data_timesteps[:,[self._timesteps - 1]]
        self.x_test, self.y_test = test_data_timesteps[:,:self._timesteps - 1], test_data_timesteps[:,[self._timesteps - 1]]

        if show_data == True:
            print(f" X Train Shape = {self.x_train.shape} \n Y Train Shape = {self.y_train.shape}")
            print(f" X Test Shape = {self.x_test.shape} \n Y Test Shape = {self.y_test.shape}")

        return self.x_train, self.y_train, self.x_test, self.y_test, self.scaler

    def svr_model(self, kernel='rbf',gamma=0.5, C=10, epsilon = 0.05):

        #Train Model
        self.model = SVR(kernel= kernel,gamma=gamma, C=C, epsilon = epsilon)
        self.model.fit(self.x_train, self.y_train[:,0]) 

        return self.model

    def predict(self):
        
        # Extracting price values of entire dataset as numpy array
        data = self._df.copy().values

        # Scaling the entire dataset based on scaler configured in train_test_split function
        data = self.scaler.transform(data)

        # Transforming to 2D tensor as per model input requirement
        data_timesteps=np.array([[j for j in data[i:i+self._timesteps]] for i in range(0,len(data)-self._timesteps+1)])[:,:,0]
        print("Full Dataset Tensor Shape: ", data_timesteps.shape)

        # Selecting inputs and outputs from data
        x = data_timesteps[:,:self._timesteps-1]
        Y = data_timesteps[:,[self._timesteps-1]]

        # Make model predictions
        Y_pred = self.model.predict(x).reshape(-1,1)

        # Inverse scale and reshape
        Y_pred = self.scaler.inverse_transform(Y_pred)
        Y = self.scaler.inverse_transform(Y)

        x_timestamps = self._df.index[self._timesteps-1:]

        plt.figure(figsize=(16,8))
        plt.plot(x_timestamps, Y, linewidth= 1.85, alpha = 0.6)
        plt.plot(x_timestamps, Y_pred, color = 'red', linewidth=1)
        plt.legend(['Actual','Predicted'])
        plt.title(f"{self._file} Prediction")
        plt.xlabel('Timestamp')

        plt.show()

        df_pred = pd.DataFrame({'Date':x_timestamps.tolist(), 'Y':Y.tolist(), 'Y_pred':Y_pred.tolist()})
        
        print(f"For {self._file} Next day Price Will be: U${Y_pred[-1,:]}")

        return Y, Y_pred, Y_pred[-1,:]
