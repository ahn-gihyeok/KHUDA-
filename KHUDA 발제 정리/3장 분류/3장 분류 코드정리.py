#3.1 MNIST
#MNIST 데이터셋 내려받기
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',as_frame=False)

#MNIST 데이터 특성 출력
X,y=mnist.data,mnist.target
print(X)
print(X.shape) #(70000,784):7만개의 데이터가 있음,28*28 픽셀임으로 784개의 특성 있음
print(y)
print(y.shape) #(70000,)

#훈련 세트,데이터 세트 나누기
X_train, X_test, y_train, y_test=X[:60000], X[60000:], y


#3.2 이진 분류기
