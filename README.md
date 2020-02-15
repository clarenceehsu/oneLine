# oneLine

 Make every step oneLine. 

## 这是什么？

提供了一些库的高级封装，旨在提供更加简单的运行模式和更简洁的代码。此库为学习和分析所用代码的集合，内容暂时不完善。

举个例子：

```python
    from oneLine import OneData, OneDatalist
    # oneLine 中的数据分为两种，一种是基于 DataFrame 类型扩展的 OneData 数据格式，一种是地址列表所代表的文件列表 OneDatalist 格式。

    data = OneData('test.csv')  # 导入数据，格式可以是文件路径或者列表、DataFrame 等

    data = data.fill_na() # 填补空缺值

    summary = data.summary() # EDA 的简单总结

    data.comparing_variables(data, 'parameter1', 'parameter2') # data 中两个参数的相关性分析并生成图

```

