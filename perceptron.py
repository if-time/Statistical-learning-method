import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


class MyPerceptron:
    def __init__(self):
        self.w = None
        self.b = 0
        self.l_rate = 1  # xuexilv

    def fit(self, x_train, y_train):
        # shape[1] 为第一维的长度，.shape[0] 为第二维的长度
        self.w = np.zeros(x_train.shape[1])
        i = 0
        while i < x_train.shape[0]:
            x = x_train[i]
            y = y_train[i]
            if y * (np.dot(self.w, x) + self.b) <= 0:
                self.w = self.w + self.l_rate * np.dot(y, x)
                self.b = self.b + self.l_rate * y
                i = 0  # 如果是误判点，从头开始检测
                # print("w", self.w)
                # print("b", self.b)
            else:
                i += 1


def draw(X, w, b):
    X_new = np.array([[0], [6]])
    # print(X_new)
    y_predict = -b - (w[0] * X_new) / w[1]
    # print(y_predict)
    plt.plot(X[:2, 0], X[:2, 1], "g*", label="1")
    plt.plot(X[2:, 0], X[2:, 0], "rx", label="-1")
    # 绘制分离超平面
    plt.plot(X_new, y_predict, "b-")
    # 设置两坐标轴起止值
    plt.axis([0, 6, 0, 6])
    # 设置坐标轴标签
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 显示图例
    plt.legend()
    # 显示图像
    plt.show()


def main():
    x_train = np.array([[3, 3], [4, 3], [1, 1]])
    y_train = np.array([1, 1, -1])
    perceptron = MyPerceptron()
    perceptron.fit(x_train, y_train)
    print(perceptron.w)
    print(perceptron.b)
    draw(x_train, perceptron.w, perceptron.b)
    # from sklearn.linear_model import Perceptron
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)
    print("w:", perceptron.coef_, "\n", "b:", perceptron.intercept_,
      "\n", "n_iter:", perceptron.n_iter_)
    res = perceptron.score(x_train, y_train)
    print("correct rate:{:.0%}".format(res))

if __name__ == "__main__":
    main()
