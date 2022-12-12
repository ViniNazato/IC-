##### Desenvolvido por Vinícius Roberto Simões Nazato Nº USP 11282009###########

import pandas as pd
import matplotlib.pyplot as plt


class SplitDataset:

    def __init__(self,file_name: str, split1_start= '2019-01-01', split1_end= '2020-12-31', split2_start = '2021-01-01', split2_end = None):
        self.file_name = file_name
        self.coin_name = file_name[:-5]
        self.split1_start = split1_start
        self.split1_end = split1_end
        self.split2_start = split2_start
        self.split2_end = split2_end
        self.ds_1, self.ds_2 = self.load_data()
       
    def load_data(self):
       
        #Load the data
        df = pd.read_excel(self.file_name, index_col=0) 
        #Set Date as index
        df = df.set_index(pd.DatetimeIndex(df['date'].values))
        #Convert 'date' column from object to datetime
        #df['date'] = pd.to_datetime(df['date'])
        #divide os datasets
        ds_1 = df[self.split1_start:self.split1_end] #dynamic
        ds_2 = df[self.split2_start:self.split2_end] #dynamic
        
        return [ds_1, ds_2]


    def show_datasets(self):

        #Datasets Shapes
        print(f"{self.coin_name} DS 1 Shape: {self.ds_1.shape}")
        print(f"{self.coin_name} DS 2 Shape: {self.ds_2.shape}")
        
        #Create lists for for loop
        dss = [self.ds_1, self.ds_2]
        model_color = ["m", "c"]

        #Create Datasets Graphs
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), sharey=True)        
        for i in range(2):
            axis[i].plot(
                dss[i]['date'],
                dss[i].iloc[:,1],
                color=model_color[i],
                label=f"Dataset {i}",
            )

            axis[i].tick_params(axis='x', labelrotation=45)

            #if max(dss[i]['BTC']) > 15000:
                #
                # axis[i].set_yticks(np.arange(min(dss[i]['BTC']), max(dss[i]['BTC']), step = 10000))
            #elif max(dss[i]['BTC']) < 10000:
                #axis[i].set_yticks(np.arange(min(dss[i]['BTC']), max(dss[i]['BTC'])+1, step = 500))   
            #axis[i].set_xlabel("Date")
            #axis[i].set_ylabel("Price")

        fig.text(0.5, 0.04, "Date", ha="center", va="center")
        fig.text(0.06, 0.5, "Price", ha="center", va="center", rotation="vertical")
        fig.suptitle(f"Dataset Division {self.coin_name}", fontsize=14)
        return plt.show()

