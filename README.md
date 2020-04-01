# oneLine

 Make every step oneLine. 

## What is it?

It's a personal code set for learning and researching, and it provides advanced encapsulation of some libraries, designing to provide a simpler mode of operation and help people to simplify their codes concisely.

> This library is a collection of code used for learning and analysis, and the content is not perfect for the time being.

For example:

```python
from oneLine import OneData
# this how we measure data in oneLine. OneData is an extension class of DataFrame.

data = OneData('test.csv')
# input the data, which can be a address, ndarray or DataFrame

data = data.fill_na()
# fill the NaN values

summary = data.summary(info=False)
# simple summary of data, which will print the summary if info stays True

data.means('parameter1', hue='parameter2')
# calculate the means, which can have a hue variable applied
```

