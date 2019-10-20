# oneLine 一行

 Make every step oneLine. 一行一步，一步一行。

## 这是什么？

这一个库封装了一些高级的接口，旨在提供更加简单的运行模式和更简洁的代码。因为此库为作者所在领域的学习分析所用代码的集合，所以内容不一定完善。

如果您也感兴趣请看下面的简单介绍，先举个例子：

举个例子：

```python
from oneLine.modules.data import *
# 这里面包含常见的库调用，如 np、plt、sns 等，一行即可调用其所有，对不同的常用库集合进行了分离。

from oneLine.io import auto_read
# 调用 io 库里面的 auto_read，这个方法可以自动识别后缀选择适合的方式从文件中导入数据。
from oneLine.dataanalysis import *
# 调用用于数据分析的 dataanalysis 模块，里面有常用的数据分析模式。

data = auto_read('test.csv')  # 导入数据

data = fill_na(data) # 填补空缺值

summary(data) # 对数据 EDA （探索性分析）的简单总结

comparing_variables(data, 'parameter1', 'parameter2') # data 中两个参数的相关性分析并生成图

```

## 更新日志

### 10.20.2019

- 优化了 ``modules`` 函数调用的部分，对不同类别的库进行了分离。现在可以选择性地批量调用常用库，减小了内存消耗和初始化的时间。（这一部分的方法还不够简洁，优化中）
- 更新 Readme.md

### 10.6.2019

- 添加更新日志
- 更新 Readme.md