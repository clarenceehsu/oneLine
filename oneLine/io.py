import numpy as np
import pandas as pd

@staticmethod
def auto_read(_dir):
    file_form = _dir.split('.')[-1]
    if file_form == 'csv':
        return pd.read_csv(_dir)
    if file_form == 'xls' or file_form == 'xlsx':
        return pd.read_excel(_dir)
    if file_form == 'json':
        return pd.read_json(_dir)
    if file_form == 'sql':
        return pd.read_sql(_dir)
    else:
        return 'Sorry, the format is not matched.'
