
class metrics:

    def mape(predictions, actuals):
            """Mean absolute percentage error"""
            return (abs(predictions - actuals) / actuals).mean()