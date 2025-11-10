import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def load_data(filepath="data/full_fundamental_dataset.csv"):
    """Loads and preprocesses the data."""
    df = pd.read_csv(filepath)
    column_list = [ # Helpful identifiers
            'gvkey', 'fyear', 'datadate', 'sic', 'exchg', 'fye', 'yearend', 'jdate', 'permno', 'permco', 'date_x', 'shrcd', 'exchcd',
             'year', 'date_y', 
              # Deflator
              'Pricet', 'Shrout_at_Pricet', 'ib', 'csho', 'prc', 'shrout',
              # 28 input variables
              'sale', 'cogs', 'xsga', 'xad', 'xrd', 'dp', 'xint', 'nopio', 'txt', 'xido', 'E', 'dvc', 'che', 'invt', 'rect', 'act', 'ppent', 
              'ivao', 'intan', 'at', 'ap', 'dlc', 'txp', 'lct', 'dltt', 'lt', 'ceq', 'oancf',
              'SALE_diff', 'COGS_diff', 'XSGA_diff', 'XAD_diff', 
              # 28 first order differences
              'XRD_diff', 'DP_diff', 'XINT_diff', 'NOPIO_diff', 'TXT_diff', 'XIDO_diff', 'E_diff', 'DVC_diff', 'CHE_diff', 'INVT_diff', 'RECT_diff', 
              'ACT_diff', 'PPENT_diff', 'IVAO_diff', 'INTAN_diff', 'A_diff', 'AP_diff', 'DLC_diff', 'TXP_diff', 'LCT_diff', 'DLTT_diff', 'LT_diff', 
              'CEQ_diff', 'CFO_diff',
              # Miscellanous
              'ret', 'retx', 'retadj', 'rf', 'spi', 'ivst', 'pstk'
              ]
    df = df[column_list]
    df_filtered = df[df['csho'] > 0]
    df_filtered['mkt_cap'] = df_filtered['Shrout_at_Pricet'] * df_filtered['Pricet']
    df_filtered['E_future'] = df_filtered.groupby('gvkey')['E'].shift(-1)
    df_filtered = df_filtered.dropna(subset=['E_future'])
    df_filtered = df_filtered.sort_values(['gvkey','year'])
    return df_filtered

def newey_west_tstat(series, lags=3):
    """Calculates Newey-West t-statistic for the mean of a series."""
    y = series.values
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return model.tvalues[0], model.params[0], model.bse[0]

def winsorize_series(s, lower=0.01, upper=0.99):
    """Winsorize a pandas Series at the given lower/upper quantiles."""
    lower_bound = s.quantile(lower)
    upper_bound = s.quantile(upper)
    return s.clip(lower=lower_bound, upper=upper_bound)

def get_input_list():
    """Returns the list of input variables."""
    return [
              'sale', 'cogs', 'xsga', 'xad', 'xrd', 'dp', 'xint', 'nopio', 'txt', 'xido', 'E', 'dvc', 'che', 'invt', 'rect', 'act', 'ppent',
              'ivao', 'intan', 'at', 'ap', 'dlc', 'txp', 'lct', 'dltt', 'lt', 'ceq', 'oancf',
              'SALE_diff', 'COGS_diff', 'XSGA_diff', 'XAD_diff',
              'XRD_diff', 'DP_diff', 'XINT_diff', 'NOPIO_diff', 'TXT_diff', 'XIDO_diff', 'E_diff', 'DVC_diff', 'CHE_diff', 'INVT_diff', 'RECT_diff',
              'ACT_diff', 'PPENT_diff', 'IVAO_diff', 'INTAN_diff', 'A_diff', 'AP_diff', 'DLC_diff', 'TXP_diff', 'LCT_diff', 'DLTT_diff', 'LT_diff',
              'CEQ_diff', 'CFO_diff'
           ]