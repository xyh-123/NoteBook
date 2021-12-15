# <center>Numpy库</center>

```python
import numpy as np 
demo=np.array([1.0,2,3,4,5]) 
print('#Dimension = ',demo.ndim) #几维数组 
print('Dimension=',demo.shape)   #数组的行列 
print('Size=',demo.size)         #有多少个元素 
print('Arryay type = ',demo.dtype)#数组的元素的类型 
print(demo)
```

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/6a4f31c628f94f4a8a72bc6ab40d7af0/clipboard.png)



```Python
print(np.random.rand(5,2))#生成5行2列的[0,1)之间的矩阵 
print(np.random.randn(5,2))#生成5行2列的[0,1)之间具有正态分布的矩阵 
print(np.arange(-10,10,2))#生成一维[-10,10]间距为2的矩阵 
print(np.arange(12).reshape(3,4))#生成3行4列[0,12)间距为2的矩阵 
print(np.linspace(0,1,10))#生成[0,1]之间数量为10个等差矩阵 
print(np.logspace(-3,3,7))#生成1*7的[log(10^-3),log(10^3)] 
print(np.zeros((2,3)))#生成2行3列全0矩阵 
print(np.ones((3,2)))#生成3行2列全1矩阵 
print(np.eye((3)))#生成3*3主对角线全1矩阵
```



![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/199b04b576964544a68ccb76aa85307c/clipboard.png)

```python
import numpy as np a=np.array([1,2,3,4,5]) 
print(a+1) 
print(a-1) 
print(a*2) 
print(a//2) 
print(a%2) 
print(a**2) 
print(1/a) 
print() 
x=np.array([2,4,6,8,10]) y=np.array([1,2,3,4,5]) 
print(x+y) print(x-y) print(x*y) print(x/y) print(x//y) print(x**y)
```



![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/7002cae27a054707aea3ccbe5801de39/clipboard.png)

**判断两个矩阵是否相等**

import numpy as np aim=np.array([[1,2,3],[4,5,6],[7,8,9]]) result=np.array([[1,2,3],[4,5,6],[7,8,9]]) print((aim==reslut).all())  #两个矩阵中的所有元素对应相等，则返回True，反之返回False。

**索引和切片**

冒号 : 的解释：如果只放置一个参数，如 [2]，将返回与该索引相对应的单个元素。如果为 [2:]，表示从该索引开始以后的所有项都将被提取。如果使用了两个参数，如 [2:7]，那么则提取两个索引(不包括停止索引)之间的项。

x=np.arange(-5,5) print(x) y=x[3:5] print(y) y[:]=1000 #同时会修改x中对应的值 print(y) print(x) z=x[3:5].copy() #这是生成一个新的numpy，改变不会改变x print(z) z[:]=500 print(z) print(x)

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/e3aa071b9716416abd1988e800e7fb3a/clipboard.png)

import numpy as np  a = np.arange(10) s = slice(2,7,2)   # 从索引 2 开始到索引 7 停止，间隔为2 print (a[s]) out:    [2  4  6] a = np.arange(10)   b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2 print(b) out:    [2  4  6]

my2dlist=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]#列表 print("my2dlist:") print(my2dlist) print(my2dlist[2]) #返回第三行 print(my2dlist[:][2])# ##TypeError: list indices must be integers or slices, not tuple #print(my2dlist[:,2]) 不可以这么写 my2darr=np.array(my2dlist)#二维数组 print("my2darr:") print(my2darr) #arr1[x,y], x相当于行数，y相当于列数 #arry[x][y]等价于[x,y],x!=':' # 只有：代表选取整列 print(my2darr[2][:]) print(my2darr[2,:]) #取第二行 print(my2darr[:][2]) #取第三列 print(my2darr[:,2]) #选取二维数组 print(my2darr[:2,2:])#取前两行，第三列到最后

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/46aa688a9aba4ee8b457d138b46d23d7/clipboard.png)

**ndarray也支持布尔索引。**

my2darr=np.arange(1,13,1).reshape(3,4) print(my2darr) divBy3=my2darr[my2darr%3==0] print(divBy3,type(divBy3)) divBy3LastRow=my2darr[2:,my2darr[2,:]%3==0] print(divBy3LastRow)

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/7fb39928b9154926893517e3f0dfbc61/clipboard.png)

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/710ac0e076384325b50be03238d42f25/clipboard.png)

**Numpy 算术和统计函数**

import numpy as np a=np.array([-1.4,0.4,-3.2,2.5,3.4]) print(a) print(np.abs(a))#取绝对值 print(np.sqrt(abs(a)))#取开方 print(np.sign(a))#取每一个元素的正负号 print(np.exp(a))#对每一项变为e^a[i] print(np.sort(a))#对矩阵进行排序

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/2039e7fafe784362b956ab67879d54c5/clipboard.png)

b=np.arange(-2,3) c=np.random.randn(5) print(b) print(c) print(np.add(b,c))#输出两个矩阵相加的结果 print(np.subtract(b,c))#输出b-c矩阵的结果 print(np.multiply(b,c))#输出两个矩阵相乘的结果 print(np.divide(b,c))#输出两个矩阵相除的结果 print(np.maximum(b,c))#输出两个矩阵对应位置最大值的结果

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/57fa52e74b664dcab9b4ae82e0c67667/clipboard.png)

import numpy as np a=np.array([-1.4,0.4,-3.2,2.5,3.4]) print(a) print('Max=',np.max(a))#最大值 print('Min=',np.min(a))#最小值 print('Average=',np.average(a))#平均数 print('Average=',np.mean(a))#平均数 print('Std deviation =',np.std(a))#计算标准差 print('sum =',np.sum(a))#计算综合

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/f9a92912ca6c4cd797baf045849981c9/clipboard.png)

numpy.std(a, axis=None, dtype=None, out=None, ddof=0) a： array_like，需计算标准差的数组 axis： int, 可选，计算标准差的轴。默认情况是计算扁平数组的标准偏差。 dtype： dtype, 可选，用于计算标准差的类型。对于整数类型的数组，缺省值为Float 64，对于浮点数类型的数组，它与数组类型相同。 out： ndarray, 可选，将结果放置在其中的替代输出数组。它必须具有与预期输出相同的形状，但如果有必要，类型(计算值的类型)将被转换。 ddof： int, 可选，Delta的自由度 功能：计算沿指定轴的标准差。返回数组元素的标准差 import numpy as np a = np.array([[1, 2], [3, 4]]) print(np.std(a))            # 计算全局标准差 1.118033988749895 print(np.std(a, axis=0))    # axis=0计算每一列的标准差 [1. 1.] print(np.std(a, axis=1))    # 计算每一行的标准差 [0.5 0.5]

**Numpy  线性代数**

**（1）diag**

功能：形成一个以一维数组为对角线元素的矩阵或输出矩阵的对角线元素

用法：np.diag(a)

a = [1,2,3]                      # 一维数组 np.diag(a) Out：array([[1, 0, 0],           [0, 2, 0],           [0, 0, 3]])           # 输出以a为对角线元素的矩阵(方阵)            b = ([1,2,3],[4,5,6],[7,8,9])    # 多维数组 np.diag(b) Out：array([1, 5, 9])            # 输出矩阵b的对角线元素

**（2）dot**

功能：计算矩阵的乘法（两个数组的内积/点积）

用法：np.dot(a,b)

a = ([1,2],[3,4]) b = ([5,6],[7,8]) np.dot(a,b) Out：array([[19, 22],            [43, 50]])

**（3）trace**

功能：计算矩阵的迹（对角线元素的和）

用法：np.trace(a)

a = ([1,2],[3,4]) np.linalg.det(a) Out：-2

**（5）eig**

功能：计算方阵的特征值和特征向量

用法：np.linalg.eig(a)

a = ([1,2],[3,4]) e,v = np.linalg.eig(a) In：e Out： array([-0.37228132,  5.37228132])      # 特征值 In：v Out： array([[-0.82456484, -0.41597356],       	     [ 0.56576746, -0.90937671]])    # 特征向量

**（6）inv**

功能：计算矩阵的逆矩阵

用法：np.linalg.inv(a)

a = ([1,2],[3,4]) np.linalg.inv(a) Out：array([[-2. ,  1. ],            [ 1.5, -0.5]])

**（7）qr**

功能：QR分解（正交分解）

用法：np.linalg.qr(a)

a = ([1,2],[3,4]) q,r = np.linalg.qr(a) In：q Out：array([[-0.31622777, -0.9486833 ],            [-0.9486833 ,  0.31622777]]) In：r Out：array([[-3.16227766, -4.42718872],         	[ 0.        , -0.63245553]])

**（8）svd**

功能：SVD分解（奇异值分解）

用法：np.linalg.svd(a,full_matrices=1,compute_uv=1)

a = ([1,2],[3,4]) u,s,v = np.linalg.svd(a) In：u Out：array([[-0.40455358, -0.9145143 ],           [-0.9145143 ,  0.40455358]]) In：s Out：array([5.4649857 , 0.36596619]) In：v Out：array([[-0.57604844, -0.81741556],           [ 0.81741556, -0.57604844]])

**（9）solve**

功能：解形如AX=B的线性矩阵方程（计算A^(-1)B的解）

用法：np.linalg.solve(a,b)

a = [[1,1,1],[ 0,2,5],[ 2,5,-1]] b = [[6],[-4],[27]] np.linalg.solve(a,b)	   Out：array([[ 5.],      		[ 3.],      		[-2.]]) 

**（10）lstsq**

功能：估计线性模型中的系数（最小二乘解，求b=a*x中的a）

用法：np.linalg.lstsq(x,b)

x = ([1,2],[3,4],[5,6],[7,8]) b = [-1,0.2,0.9,2.1] np.linalg.lstsq(x,b) Out：(array([ 1.95, -1.45]),   array([0.05]),    2,    array([14.2690955 ,  0.62682823]))

**Pandas ——Series**

**Pandas 中****整型为int64，浮点型为float64****，字符串、布尔型等****其他数据类型为object**

import pandas as pd pd.read_csv(filepath_or_buffer,header,parse_dates,index_col) 参数： filepath_or_buffer： 字符串，或者任何对象的read()方法。这个字符串可以是URL， 有效的URL方案包括http、ftp、s3和文件。可以直接写入"文件名.csv" header： 将行号用作列名，且是数据的开头。 注意当skip_blank_lines=True时，这个参数忽略注释行和空行。 所以header=0表示第一行是数据而不是文件的第一行。 【注】：如果csv文件中含有中文，该如何？ 1、可修改csv文件的编码格式为unix(不能是windows)（用notepad++打开修改） 2、df = pd.read_csv(csv_file, encoding="utf-8")， 设置读取时的编码或 encoding="gbk" 3、在使用列名来访问DataFrame里面的数据时， 对于中文列名，应该在列名前面加'u'， 表示后面跟的字符串以unicode格式存储，如下所示 print(df[u"经度(度)"]) (1)、header=None 即指定原始文件数据没有列索引，这样read_csv为其自动加上列索引{从0开始} ceshi.csv原文件内容： c1,c2,c3,c4 a,0,5,10 b,1,6,11 c,2,7,12 d,3,8,13 e,4,9,14 df=pd.read_csv("ceshi.csv",header=None) print(df) 结果：    0   1   2   3 0  c1  c2  c3  c4 1   a   0   5  10 2   b   1   6  11 3   c   2   7  12 4   d   3   8  13 5   e   4   9  14 (2)、header=None，并指定新的索引的名字names=seq序列 df=pd.read_csv("ceshi.csv",header=None,names=range(2,6)) print(df) 结果：    2   3   4   5 0  c1  c2  c3  c4 1   a   0   5  10 2   b   1   6  11 3   c   2   7  12 4   d   3   8  13 5   e   4   9  14  (3)、header=None，并指定新的索引的名字names=seq序列；如果指定的新的索引名字的序列比原csv文件的列数少，那么就截取原csv文件的倒数列添加上新的索引名字 df=pd.read_csv("ceshi.csv",header=0,names=range(2,4)) print(df) 结果：        2   3 c1 c2  c3  c4 a  0    5  10 b  1    6  11 c  2    7  12 d  3    8  13 e  4    9  14  (4)、header=0 表示文件第0行（即第一行，索引从0开始）为列索引 df=pd.read_csv("ceshi.csv",header=0) print(df) 结果：  c1  c2  c3  c4 0  a   0   5  10 1  b   1   6  11 2  c   2   7  12 3  d   3   8  13 4  e   4   9  14 (5)、header=0，并指定新的索引的名字names=seq序列 df=pd.read_csv("ceshi.csv",header=0,names=range(2,6)) print(df) 结果：   2  3  4   5 0  a  0  5  10 1  b  1  6  11 2  c  2  7  12 3  d  3  8  13 4  e  4  9  14 注：这里是把原csv文件的第一行换成了range(2,6)并将此作为列索引 (6)、header=0，并指定新的索引的名字names=seq序列；如果指定的新的索引名字的序列比原csv文件的列数少，那么就截取原csv文件的倒数列添加上新的索引名字 df=pd.read_csv("ceshi.csv",header=0,names=range(2,4)) print(df) 结果：     2   3 a 0  5  10 b 1  6  11 c 2  7  12 d 3  8  13 e 4  9  14  parse_dates： 布尔类型值 or int类型值的列表 or 列表的列表 or 字典（默认值为 FALSE） (1)True:尝试解析索引 (2)由int类型值组成的列表(如[1,2,3]):作为单独数据列，分别解析原始文件中的1,2,3列 (3)由列表组成的列表(如[[1,3]]):将1,3列合并，作为一个单列进行解析 (4)字典(如{'foo'：[1, 3]}):解析1,3列作为数据，并命名为foo  index_col： int类型值，序列，FALSE（默认 None） 将真实的某列当做index（列的数目，甚至列名） index_col为指定数据中那一列作为Dataframe的行索引，也可以可指定多列，形成层次索引，默认为None,即不指定行索引，这样系统会自动加上行索引。 举例： df=pd.read_csv("ceshi.csv",index_col=0) print(df) 结果：    c2  c3  c4 c1             a    0   5  10 b    1   6  11 c    2   7  12 d    3   8  13 e    4   9  14 表示：将第一列作为索引index df=pd.read_csv("ceshi.csv",index_col=1) print(df) 结果：   c1  c3  c4 c2            0   a   5  10 1   b   6  11 2   c   7  12 3   d   8  13 4   e   9  14 表示：将第二列作为索引index  df=pd.read_csv("ceshi.csv",index_col="c1") print(df) 结果：    c2  c3  c4 c1             a    0   5  10 b    1   6  11 c    2   7  12 d    3   8  13 e    4   9  14 表示：将列名"c1"这里一列作为索引index 【注】：这里将"c1"这一列作为索引即行索引后，"c1"这列即不在属于列名这类，即不能使用df['c1']获取列值 【注】：read_csv()方法中header参数和index_col参数不能混用，因为header指定列索引，index_col指定行索引，一个DataFrame对象只有一种索引 squeeze： 布尔值，默认FALSE TRUE 如果被解析的数据只有一列，那么返回Series类型。

**Series定义**

- **Series像是一个Python的****dict类型****，因为它的索引与元素是映射关系**
- **Series也像是一个ndarray类型，因为它也可以通过series_name[index]方式访问**
- **Series是****一维****的，但能够存储不同类型的数据**
- **每个Series都有一组索引与数据对应，若不指定则默认为整型索引**

\# Series 默认索引（不显式指定index，则Series使用默认索引，[0, 1, 2, 3, 4...] series1 = pd.Series([10, 7, -4, 1]) # 或者通过以下方式创建Series l = [10, 7, -4, 1] series1 = pd.Series(l)

out:    series1 0    10 1     7 2    -4 3     1 dtype: int64

series2 = pd.Series(['ant', 'bear', 'cat', 'dog'], index=['a', 'b', 'c', 'd']) # 或者通过以下方式创建Series l = ['ant', 'bear', 'cat', 'dog'] index = ['a', 'b', 'c', 'd'] series1 = pd.Series(l, index=index)

out:    series2 a     ant b    bear c     cat d     dog dtype: object

相比于python中的dict，Series中索引与元素是一种映射关系，元素在Series对象中是有序存储的，并是通过索引实现其有序的。

如果python版本 >= 3.6 并且 Pandas 版本 >= 0.23 , 则通过dict创建的Series索引按照dict的插入顺序排序 如果python版本 < 3.6 或者 Pandas 版本 < 0.23， 则通过dict创建的Series索引按照按词汇顺序排列

指定**dict**索引顺序创建**Series**

\#对于指定索引names未出现的index ’d’ ，则自动过滤掉 d = {'b': 1, 'a': 0, 'c': 2, 'd': 3} names = ['a', 'b', 'c'] series4 = pd.Series(d,index=names)

out:    series4 a    0 b    1 c    2 dtype: int64

\#若names中出现dict中没有的索引，则该索引对应值为NaN d = {'b': 1, 'a': 0, 'c': 2, 'd': 3} names = ['a', 'b', 'c' ,'e'] series4 = pd.Series(d,index=names)

out:    series4 a    0.0 b    1.0 c    2.0 e    NaN dtype: float64

通过ndarray创建

series5 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e']) # 或者通过以下方式创建Series l = np.random.randn(5) index = ['a', 'b', 'c', 'd', 'e'] series1 = pd.Series(l, index=index)

out:    series5 a    1.025255 b   -0.684486 c    1.870848 d   -0.517517 e   -0.087879 dtype: float64

通过标量创建

series6 = pd.Series(5, index=['a', 'b', 'c']) series7 = pd.Series(5.0, index=['a', 'b', 'c']) # 对于创建float64来说，可缩写标量，eg： series7 = pd.Series(5., index=['a', 'b', 'c']) out:    series6 a    5 b    5 c    5 dtype: int64    series7 a    5.0 b    5.0 c    5.0 dtype: float64

**Series元素的访问**

**series_name[****index****]** 方式，这里的index指的是在不给Series显式指定index的时候，Series默认的整型索引

series1 = pd.Series([10, 7, -4, 1]) series2 = pd.Series(['ant', 'bear', 'cat', 'dog'], index=['a', 'b', 'c', 'd']) # 或者通过以下方式创建Series l = ['ant', 'bear', 'cat', 'dog'] index = ['a', 'b', 'c', 'd'] series2 = pd.Series(l, index=index) # out:ant print(series2[0]) #只输出索引对应的值 # out:a      ant;b     bear;c    camel;dtype: objectd print(series2[:3]) #输出前三个键值对 # out:0    10;1     7;dtype: int64 print(series1[series1 > series1.median()])  # out:c    camel;d      dog;b     bear;dtype: object print(series2[[2, 3, 1]]) #按照对应的索引返回键值对

**series_name[key]，这里的key指的是在给 Series 显式指定的index**

print(series2['a']) # out:ant

　使用 loc 或者 iloc （切片）查看数据值，区别是 loc 是根据行名，iloc 是根据数字索引：

　　①loc：(location)，works on labels in the index，只能使用字符型标签来索引数据，不能使用数字来索引数据，不过有特殊情况，当数据框dataframe的行标签或者列标签为数字，loc就可以来其来索引。

　　②iloc：(i=integer)，works on the positions in the index (so it only takes integers)，主要使用数字来索引数据，而不能使用字符型的标签来索引数据。

import pandas as pd import numpy as np data = pd.Series(np.arange(10), index=[49,48,47,46,45, 1, 2, 3, 4, 5]) print('data:\n',data,'\n') print('data.iloc[:3]:\n',data.iloc[:3],'\n') print('data.loc[:3]:\n',data.loc[:3],'\n') print('data.ix[:3]:\n',data.ix[:3],'\n') data: 49    0 48    1 47    2 46    3 45    4 1     5 2     6 3     7 4     8 5     9 dtype: int64  data.iloc[:3]: 49    0 48    1 47    2 dtype: int64  data.loc[:3]: 49    0 48    1 47    2 46    3 45    4 1     5 2     6 3     7 dtype: int64  data.ix[:3]: 49    0 48    1 47    2 46    3 45    4 1     5 2     6 3     7 dtype: int64 

**Series内容修改**

Series元素值的修改

series2['c'] = 'camel' # 或者 series2['2'] = 'camel' print(series2['c'])   # out:camel

Series元素索引的修改

\# 通过series.index 可以获取到Series的索引，替换该索引即可 print(series2.index)  # Index(['a', 'b', 'c', 'd'], dtype='object') # series.index 是一个list对象，可通过series.index[index]来访问指定的索引并替换之 series.index[index]="新索引"

**Series的元素属性**

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/3ce60d96845e48e8b30d4128ee4d54a2/clipboard.png)

series.shape #查询Series有多少行

**Series常用函数**

\#如果不指定deep参数，则默认deep=True #deep参数设置为True，则实现深拷贝，创建一个新对象，对series进行复制 cpys = series2.copy(deep=True)  print(cpys.values is series2.values or cpys.index is series2.index) # False """ deep参数设置为False，则实现浅拷贝，创建一个新对象， 但不复制原series的数据，也不复制其索引，仅对索引与数据指向原数据， 不同于（cpys = series2) """ cpys = series2.copy(deep=False) cpys2 = series2   # 该操作不创建对象，只对原对象创建一个新的变量名称

**Series**重设索引**reindex**函数

reindex() 函数会创建一个新的对象，用以适应新的索引，并不会修改源对象

s = pd.Series(['Tom', 'Kim', 'Andy'], index=['No.1', 'No.2', 'No.3']) rs = s.reindex(['No.0', 'No.1', 'No.2', 'No.3', 'No.4']) # 缺失索引对应数值默认使用Nan填充 rs2 = s.reindex(['No.0', 'No.1', 'No.2', 'No.3', 'No.4'], fill_value='填充值') # 设置索引对应数值默认使用“填充值”填充 out:    rs No.0     NaN No.1     Tom No.2     Kim No.3    Andy No.4     NaN dtype: object 	rs2 No.0     填充值 No.1     Tom No.2     Kim No.3    Andy No.4     填充值 dtype: object

method参数

- **ffill**或**pad**：前向填充，即将缺失值的前一个索引的值填充在缺失值位置上
- **bfill**或**backfill**：*后向（或进位）填充*，即将缺失值的后一个索引的值填充在缺失值位置

s = pd.Series(['Tom', 'Kim', 'Andy'], index=['No.1', 'No.2', 'No.3']) rs = s.reindex(['No.0', 'No.1', 'No.4', 'No.5'], method='ffill') rs2 = s.reindex(['No.0', 'No.1', 'No.4', 'No.5'], method='bfill') out:    rs No.0     NaN    # 由于前一个索引没有值，则不填充 No.1     Tom No.4    Andy    # 因为前向填充(取No.3的值Andy作为填充值) No.5    Andy    # 取No.4的值作为填充值 dtype: object        rs2 No.0    Tom No.1    Tom     No.4    NaN     # 取No.5的值作为填充值，即NaN No.5    NaN     # 由于后一个索引没有值，则不填充，默认为NaN dtype: object

**Series** 删除元素

series2 = pd.Series(['ant', 'bear', 'cat', 'dog'], index=['a', 'b', 'c', 'd']) # 或者通过以下方式创建Series series2.drop('c') print(series2) series2 a      ant b     bear c    camel d      dog dtype: object

**Pandas——DataFrame**

DataFrame 是 Python 中 Pandas 库中的一种数据结构，是一种二维表。

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/35870de1b46d4e8ebecf2df721230de8/clipboard.png)

\#由字典生成 cars={    'make':['Ford','Honda','Toyota','Tesla'],    'model':['Taurus','Accord','Camry','Model S'],    'MSPR':[27595,23570,23495,68000], } carData=DataFrame(cars) print("carData:") print(carData) print("carData.index:") #此时是默认索引数字 print(carData.index) print("\ncarData.columns:")#列名列表 print(carData.columns)

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/f5e414e434d74edbb0a87e6df31373d0/clipboard.png)

carData2=DataFrame(cars,index=['NO.1','NO.2','NO.3','NO.4'])#改变索引 carData2['year']=2018   #添加新列 carData2['dealership']=['Courtesy Ford','Capital Honda','Spartan Toyota','N/A'] print("\ncarData2:") print(carData2) print("carData2.iloc[1,2] = ",end="") print(carData2.iloc[1,2])  #取数字行为1,第三列的值 print("carData2.loc['NO.1','model']=",end="") # print(carData2.loc['NO.1','model']) print('carData2.iloc[1:3,1:3]=') #取2,3行， print(carData2.iloc[1:3,1:3]) print('carData2.shape=',carData2.shape) print('carData2.size=',carData2.size) print('carData2[carData2.MSPR>25000]:') print(carData2[carData2.MSPR>25000])

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/5c9281121d6e41938b8e5f37fc12dedd/lhq)eq8~nkxb.png)

从元组列表生成

tuplelist=[(2011,45.1,32.4),(2012,42.4,34.5),(2013,47.2,32.9),           (2014,44.2,31.4),(2015,39.9,29.8),(2016,41.5,36.7)] columnNames=['year','temp','precip'] weatherData=DataFrame(tuplelist,columns=columnNames)#由元组生成DataFrame print("\nweatherData") print(weatherData)

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/76d3bc7bcfc642bb99891ddc0fcb863b/clipboard.png)

numpy生成DataFrame

npdata=np.random.randn(5,3) columnNames1=['x1','x2','x3'] data=DataFrame(npdata,columns=columnNames1) print("\ndata:") print(data) print('Data transpose operation:') print(data.T)       #将行列交换 print("Addition:") print(data.values+4) print('Multiplication:') print(data.values*10) print("data['x2']=") print(data['x2']) print("type(data[\'x2\'])") print(type(data['x2'])) print('Row 3 of data table:') print(data.iloc[2]) #输出第二行的table print(type(data.iloc[2])) print("Row 3 of car data table:") print(carData2.iloc[2])

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/2301c8c30bba4fd0bdef126d469ad45d/clipboard.png)

进行算数运算

print(data) data2=DataFrame(np.random.randn(5,3),columns=columnNames1) print('\ndata2') print(data2) print("\ndata+data2=") print(data.add(data2)) print("\ndata*data2=") print(data.mul(data2))

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/8501088d8a48458aa9da020a555980d7/clipboard.png)

**统计运算**

describe()函数可以方便地查看每一个列的常见统计量

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/c7f4fdf775334884a516be0b1a712eaf/clipboard.png)

import pandas as pd data=pd.read_csv("C://Users//xyh//PycharmProjects//untitled2//爬虫//iris.data",header=None)#读CSV文件 print(data.describe())# data.head(n)#输出前n行，默认为前5行

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/950803a762d747b3b0b24e2e97f39ec5/clipboard.png)

读取文件

import pandas as pd data=pd.read_csv(path,header='infer')

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/460ee0f72178431ba5202c1b719c1112/clipboard.png)

\#data['Class']中列的值替换 data['Class']=data['Class'].replace( ['fishes','birds','amphibians','reptiles'],'non-mammals')

**crosstab函数**

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/6651c7c384f84e35851529b6d4ede4ca/clipboard.png)

pd.crosstab(df.Sex, df.Handedness, margins = True) 第一个参数是列, 第二个参数是行. 还可以添加第三个参数: 列和行都可以是列表     pd.crosstab( [data['Warm-blooded'],data['Gives Birth']],data['Class']) """ 第一列 """

![img](C:/Users/xyh/AppData/Local/YNote/data/qq93579132A5C48C293AB071F1C2AE08F5/8adffc1c247a403db4ecc3f2cdc0eff1/clipboard.png)

**pandas中的get_dummies方法**

pandas.get_dummies(data, prefix=None, prefix_sep='_',  dummy_na=False, columns=None, sparse=False, drop_first=False)

data : array-like, Series, or DataFrame

输入的数据

prefix : string, list of strings, or dict of strings, default None

get_dummies转换后，列名的前缀

columns : list-like, default None

指定需要实现类别转换的列名

dummy_na : bool, default False

增加一列表示空缺值，如果False就忽略空缺值

drop_first : bool, default False

获得k中的k-1个类别值，去除第一个