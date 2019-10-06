# oneLine 一行

 Make every step oneLine. 一行一步，一步一行。

## 这是什么？

这一个库封装了一些高级的接口，旨在提供更加简单的运行模式和更简洁的代码。因为此库主要用于作者所在领域的学习分析，所以内容不一定完善。

举个例子：

```python
from oneLine.modules import *
# 这里面包含常见的库调用，如 np、plt、sns 等，一行即可调用其所有（后续会对不同的行业或者领域常用库进行分隔）。

from oneLine.io import auto_read
# 调用 io 库里面的 auto_read，这个方法可以自动识别后缀选择适合的方式从文件中导入数据。
from oneLine.dataanalysis import *
# 调用用于数据分析的 dataanalysis 模块，里面有常用的数据分析模式。

data = auto_read('test.csv')  # 导入数据

data = fill_na(data) # 填补空缺值

summary(data) # 对数据 EDA （探索性分析）的总结

comparing_variables(data, 'parameter1', 'parameter2') # data 中两个参数的相关性分析并输出相关图

```

## 更新日志

### 10.6.2019

- 添加更新日志
- 更新 Readme.md