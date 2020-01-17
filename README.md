# oneLine 一行

 Make every step oneLine. 一行一步，一步一行。

## 这是什么？

这一个库封装了一些高级的接口，旨在提供更加简单的运行模式和更简洁的代码。因为此库为作者所在领域的学习分析所用代码的集合，所以内容不一定完善。

如果您也感兴趣请看下面的简单介绍，先举个例子：

举个例子：

```python
    from oneLine import OneData, OneDatalist
    # oneLine 中的数据分为两种，一种是基于 DataFrame 类型扩展的 OneData 数据格式，一种是地址列表所代表的文件列表 OneDatalist 格式。

    data = OneData('test.csv')  # 导入数据，格式可以是文件路径或者列表、DataFrame等

    data = data.fill_na() # 填补空缺值

    summary = data.summary() # 对数据 EDA （探索性分析）的简单总结

    comparing_variables(data, 'parameter1', 'parameter2') # data 中两个参数的相关性分析并生成图

```

## 更新日志

### 1.17.2020

- 改进了文件结构，增加了 Exception

### 12.2.2019

- 修改了 OneDatalist 的数据导入方式，感觉目前的功能还较为孱弱
- 改进了曲线平滑算法，将已弃用的 spline 改为了 interp1d
- 版本号更新为 1.0.0，更新日志简化

### 11.20.2019

- 整合并删除了多余的文件，添加并完善了 docstring
- 部分功能改进

### 11.11.2019

- 增加了 Machine Learning 的部分算法，可以耦合 OneData 进行使用
- 修复部分 bug

### 11.09.2019

- 修复了一些 OneData 输入数据时的 bug
- 增加了数据内存消耗优化的算法
- 正有想法去做新的深度学习部分的 OneLine，从而方便以后能够通过 OneLine 更方便地生成神经网络并调试

### 11.04.2019

- 改进的算法，修复了一些小 bug
- 优化了结构，添加了一些功能，现在使用更方便了
- 添加了测试部分，方便使用者进行测试

### 10.20.2019

- 优化了 ``modules`` 函数调用的部分，对不同类别的库进行了分离。现在可以选择性地批量调用常用库，减小了内存消耗和初始化的时间。（这一部分的方法还不够简洁，优化中）
- 更新 Readme.md

### 10.6.2019

- 添加更新日志
- 更新 Readme.md