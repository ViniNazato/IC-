##### Desenvolvido por Vinícius Roberto Simões Nazato Nº USP 11282009###########

import pandas as pd 
from pycoingecko import CoinGeckoAPI # importa o SDK # barra de progresso
from datetime import datetime, timezone


class GeckoAPI():

    def __init__(self, strt_date:str, end_date:str, id:str, symbol:str, name:str, crncy='usd'):
        
        self._strt_date = str(int(datetime.strptime(strt_date,"%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())) #string -> Timestamp transformation to miliseconds
        self._end_date = str(int(datetime.strptime(end_date ,"%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())) #Timestamp transformation to miliseconds
        self._id = id
        self._symbol = symbol
        self._name = name
        self._crncy = crncy
        self._cg = CoinGeckoAPI() 
        pass
    
    @property
    def get_strtdate(self):
        
        return self._strt_date
    
    #mais properties

    def api_call(self):

        coin_data = self._cg.get_coin_market_chart_range_by_id(
            id = self._id,
            symbol = self._symbol,
            name = self._name,
            vs_currency = self._crncy,
            from_timestamp = self._strt_date,
            to_timestamp = self._end_date            
        )

        df = pd.DataFrame(coin_data['prices'])
        df.columns = ['date',self._symbol.upper()]
        df['date'] = df['date'].apply(lambda i: datetime.fromtimestamp(i/1000.0).strftime("%Y-%m-%d"))

        return df
    
