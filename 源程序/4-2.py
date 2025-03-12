from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score
iris=load_iris()
iris_data=iris['data']
iris_target=iris['target']
for i in range(2,7):
kmeans=KMeans(n_clusters=i,random_state=123).fit(iris_data)
#将聚类的结果与样本真实的类别标签进行比较
    score=fowlkes_mallows_score(iris_target,kmeans.labels_)
print('聚%d类FMI评估分值为：%f' %(i,score))
