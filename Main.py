##### Desenvolvido por Vinícius Roberto Simões Nazato Nº USP 11282009###########
##### Developed by Vinícius Roberto Simões Nazato Nº USP 11282009###########
from SVRegression import SVRegression
from Utils import metrics
from CoinGeckoAPI import GeckoAPI
from SplitDataset import SplitDataset
from tqdm import tqdm
import os

############################## INPUT SECTION ###################################
#Setup start and end date - Format Ex: 2022-09-30

strt_date = '2018-01-01'
end_date = '2022-09-30'

#Ouput path

out_path_raw = r"C:\Users\vnaza\OneDrive\Documentos\IC\raw_data\\"
out_path_split = r"C:\Users\vnaza\OneDrive\Documentos\IC\split_data\\"

#This the default on SplitDataset() Class
# split1_start = '2019-01-01' 
# split1_end = '2020-12-31'
# split2_start = '2021-01-01'
# split2_end = None

#List the selected Cryptos for Analysis

list_id =     ['binancecoin', 'bitcoin']
list_symbol = [x.lower() for x in ["BNB", "BTC"]]
list_name =   [x.upper() for x in ['BNB', 'Bitcoin']]
list_file_name = []
list_dfs = []

'''
list_id = ['binancecoin','cardano','ethereum','litecoin','tron','bitcoin','ripple','dogecoin']
list_symbol = [x.lower() for x in ["BNB","ADA","ETH","LTC","TRX","BTC","XRP","DOGE"]]
list_name = ['BNB','Cardano','Ethereum','Litecoin','TRON','Bitcoin','XRP','Dogecoin']
list_dfs = []
'''

print('************************ Aquiring Data for Selected Cryptos: ****************************')
#Get Crypto prices data and sends it to Excel format
for id, symbol, name in zip(list_id,list_symbol,list_name):
    
    crypto_price = GeckoAPI(strt_date, end_date, id, symbol, name, 'usd') #
    crypto_price = crypto_price.api_call()
    print(f'{name} Coin Shape Values: {crypto_price.shape}')
    crypto_price.to_excel(str(out_path_raw + name + ".xlsx")) #Sends excel file directly to open workspace
    list_file_name.append(name + ".xlsx")
    list_dfs.append(crypto_price)

#Merge the selected

# from functools import reduce
# df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
#                                             how='outer'),list_dfs)

# df_merged.to_csv("Crypto_prices.csv")
# print(df_merged.shape)

print('************************ Data for Selected Cryptos Aquired ****************************')
print('************************ Spliting Data: ****************************')

#Calls to class to split the selected data in two

for filename in tqdm(list_file_name):
    # The user can set the Start and End Date for both Splits
    split_ds = SplitDataset(str(out_path_raw + filename)) 
    #split_ds.show_datasets() #Show the divided data
    ds_1, ds_2 = split_ds.load_data()
    ds_1.to_excel(out_path_split + 'DS_1_'+filename)
    ds_2.to_excel(out_path_split + 'DS_2_'+filename)

print('************************ Data for Selected Crytos Divided ****************************')
print('********************** Predicting Prices Based on SVR Model: *************************')

for filename in os.listdir('split_data'):
    prediction = SVRegression('split_data\\' + filename, timesteps = 5)
    prediction.train_test_split(show_data= False)
    prediction.svr_model(kernel='rbf',gamma=0.5, C=10, epsilon = .05)
    actual_price_array, predicted_price_array, predicted_price = prediction.predict()
    print(f'Filename: {filename} -> Mean absolute percentage error: {metrics.mape(predicted_price_array[:-1, :], actual_price_array[:-1, :]) * 100} %')