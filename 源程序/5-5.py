import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB		#导入高斯朴素贝叶斯
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
X ,Y= [],[]     #读取数据
fr = open("D:\\knn.txt")
for line in fr.readlines():
    line = line.strip().split()
    X.append([int(line[0]),int(line[1])])
    Y.append(int(line[-1]))
X=np.array(X)  #转换成numpy数组,X是特征属性集
Y=np.array(Y)  #y是类别标签集
#归一化
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)
# 划分训练集和测试集，测试集比例16%
train_X,test_X,train_y,test_y=train_test_split(X, Y, test_size=0.16) 
# 训练贝叶斯分类模型
model = GaussianNB()
model.fit(train_X, train_y)
print(model)		#输出模型的参数
expected = test_y			#实际类别值
predicted = model.predict(test_X) 	#预测的类别值
print(metrics.classification_report(expected, predicted))       # 输出分类信息
label = list(set(Y))    # 去重复，得到标签类别
print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵
