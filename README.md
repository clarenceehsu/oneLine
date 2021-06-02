# oneLine

Make every step oneLine. 

## What is it?

It's a personal code set for learning and researching on Data Science and Machine Learning. It contains advanced encapsulation of some libraries, designing to provide a simpler mode of operation and help people to simplify their codes concisely.

> This library is a collection of code used for learning and analysis, and the content is not perfect for the time being.

## Installation

You can install this module with command below:

```
pip install one-line
```

## How to use it?

Import module and dataset:

```python
# import OneData, an extended DataFrame
from oneline import OneData

# input the data with address, and the format will be set automatically
data = OneData('test.csv')
```

You can also import data using Pandas:

```python
import pandas as pd
from oneline import OneData

temp = pd.read_csv('test.csv')

# a DataFrame type data is also acceptable
data = OneData(temp)
```

More example see also `test/test_EDA.py`.
