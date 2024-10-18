import numpy as np
import matplotlib.pyplot as plt


def mean_normalization(x):
    N = len(x)
    maxi = np.max(x)
    mini = np.min(x)
    avg = np.mean(x)
    x = (x - avg) / (maxi - mini)
    x_b = np.c_[np.ones((N, 1)), x]  # add bias term column to x
    return x_b, maxi, mini, avg


class LinearRegression:
    def predict(self, x, thetas):
        return x.dot(thetas)

    def compute_loss(self, y_hat, y):
        return (y_hat - y) ** 2 / 2

    def compute_loss_gradient(self, y_hat, y):
        return y_hat - y

    def compute_gradient(self, x, gl):
        return x.T.dot(gl)

    def update_weight(self, thetas, gradient, lr):
        return thetas - lr * gradient

    def stochastic_gradient_descent(self, x_b, y, n_epochs=50, learning_rate=0.00001):
        # initialize thetas
        # thetas = np.random.randn(4, 1)  # uncomment this code for real application
        thetas = np.asarray([[1.16270837], [-0.81960489],
                            [1.39501033],  [0.29763545]])

        thetas_path = [thetas]  # list to store thetas for each sample
        losses = []  # list to store loss for each sample

        for _ in range(n_epochs):
            for i in range(N):
                # Select random number in N
                # random_index = np.random.randint(N) # uncomment this code for real application
                random_index = i  # This code is used for this assignment only
                x_i = x_b[random_index:random_index+1]
                y_i = y[random_index:random_index+1]

                # Compute output of sample i
                y_hat_i = self.predict(x_i, thetas)

                # Compute loss l_i of sample i
                l_i = self.compute_loss(y_hat_i, y_i)

                # Compute gradient for loss of sample i
                gl_i = self.compute_loss_gradient(y_hat_i, y_i)

                # Compute gradient of sample i
                gradient_i = self.compute_gradient(x_i, gl_i)

                # Update thetas
                thetas = self.update_weight(thetas, gradient_i, learning_rate)

                # Logging
                thetas_path.append(thetas)
                losses.append(l_i[0][0])

        return thetas_path, losses

    def mini_batch_gradient_descent(self, x_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.00001):
        # initialize thetas
        # thetas = np.random.randn(4, 1)  # uncomment this code for real application
        thetas = np.asarray([[1.16270837], [-0.81960489],
                            [1.39501033],  [0.29763545]])

        thetas_path = [thetas]  # list to store thetas for each mini-batch
        losses = []  # list to store loss for each mini-batch

        for _ in range(n_epochs):
            # shuffled_indices = np.random.permutation(N)   # uncomment this code for real application
            shuffled_indices = np.asarray([21, 144, 17, 107, 37, 115, 167, 31, 3, 132, 179, 155, 36, 191, 182, 170, 27, 35, 162, 25, 28, 73, 172, 152, 102, 16, 185, 11, 1, 34, 177, 29, 96, 22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126, 165, 78, 151, 104, 110, 53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 190, 169, 149, 79, 138, 20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 26, 8, 131, 77, 80, 130, 127, 125, 61, 10, 175, 143, 87, 33, 50, 54, 97, 9, 84, 188, 139, 195,
                                          72, 64, 194, 44, 109, 112, 60, 86, 90, 140, 171, 59, 199, 105, 41, 147, 92, 52, 124, 71, 197, 163, 98, 189, 103, 51, 39, 180, 74, 145, 118, 38, 47, 174, 100, 184, 183, 160, 69, 91, 82, 42, 89, 81, 186, 136, 63, 157, 46, 67, 129, 120, 116, 32, 19, 187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40, 83, 24, 168, 150, 178, 49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65, 176, 101, 55, 133, 13, 106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2, 134, 56, 123, 122, 154])
            x_b_shuffled = x_b[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            for i in range(0, N, minibatch_size):
                x_i = x_b_shuffled[i:i+minibatch_size]
                y_i = y_shuffled[i:i+minibatch_size]

                # Compute output of mini-batch i
                y_hat_i = self.predict(x_i, thetas)

                # Compute loss l_i of mini-batch i
                l_i = self.compute_loss(y_hat_i, y_i)

                # Compute average gradient for loss of mini-batch i
                gl_i = self.compute_loss_gradient(y_hat_i, y_i)/minibatch_size

                # Compute average gradient of mini-batch i
                gradient_i = self.compute_gradient(x_i, gl_i)

                # Update thetas
                thetas = self.update_weight(thetas, gradient_i, learning_rate)

                # Logging
                thetas_path.append(thetas)
                loss_mean = np.sum(l_i) / minibatch_size
                losses.append(loss_mean)

        return thetas_path, losses

    def batch_gradient_descent(self, x_b, y, n_epochs=50, learning_rate=0.00001):
        # initialize thetas
        # thetas = np.random.randn(4, 1)  # uncomment this code for real application
        thetas = np.asarray([[1.16270837], [-0.81960489],
                            [1.39501033],  [0.29763545]])

        thetas_path = [thetas]  # list to store thetas for each dataset pass (epoch)
        losses = []  # list to store loss for each dataset pass (epoch)

        for _ in range(n_epochs):
            x_i = x_b[:]
            y_i = y[:]

            # Compute output of dataset pass (epoch) i
            y_hat_i = self.predict(x_i, thetas)

            # Compute loss l_i of dataset pass (epoch) i
            l_i = 2 * self.compute_loss(y_hat_i, y_i)   # unlike SGD and MBGD, the loss function used by BGD in this case is (y_hat - y) ** 2

            # Compute average gradient for loss of dataset pass (epoch) i
            gl_i = 2 * self.compute_loss_gradient(y_hat_i, y_i)/N   # unlike SGD and MBGD, the loss function used by BGD in this case is (y_hat - y) ** 2

            # Compute average gradient of dataset pass (epoch) i
            gradient_i = self.compute_gradient(x_i, gl_i)

            # Update thetas
            thetas = self.update_weight(thetas, gradient_i, learning_rate)

            # Logging
            thetas_path.append(thetas)
            loss_mean = np.sum(l_i) / N
            losses.append(loss_mean)

        return thetas_path, losses


if __name__ == "__main__":
    # dataset
    data = np.genfromtxt('../Data/advertising.csv',
                         delimiter=',', skip_header=1)
    N = data.shape[0]   # number of samples
    X = data[:, :3]  # list of feature values of the samples
    y = data[:, 3:]  # list of target values of the samples

    # Normalize input data by using mean normalizaton
    X_b, maxi, mini, avg = mean_normalization(X)

    # Create an instance of LinearRegression
    linear_regression = LinearRegression()

    # -------------------- Stochastic Gradient Descent --------------------
    sgd_theta, losses = linear_regression.stochastic_gradient_descent(
        X_b, y, n_epochs=50, learning_rate=0.01)
    # Visualize the loss
    x_axis = list(range(500))
    plt.plot(x_axis, losses[:500], color="r")
    plt.show()

    # Question 1:
    sgd_theta, losses = linear_regression.stochastic_gradient_descent(
        X_b, y, n_epochs=1, learning_rate=0.01)
    print(round(np.sum(losses), 2))  # 6754.64

    # -------------------- Mini-batch Gradient Descent --------------------
    mbgd_theta, losses = linear_regression.mini_batch_gradient_descent(
        X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01)
    # Visualize the loss
    x_axis = list(range(200))
    plt.plot(x_axis, losses[:200], color="r")
    plt.show()

    # Question 2:
    mbgd_theta, losses = linear_regression.mini_batch_gradient_descent(
        X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01)
    print(round(np.sum(losses), 2))  # 8865.65

    # -------------------- Batch Gradient Descent --------------------
    bgd_theta, losses = linear_regression.batch_gradient_descent(
        X_b, y, n_epochs=100, learning_rate=0.01)
    # Visualize the loss
    x_axis = list(range(100))
    plt.plot(x_axis, losses[:100], color="r")
    plt.show()

    # Question 3:
    bgd_theta, losses = linear_regression.batch_gradient_descent(
        X_b, y, n_epochs=100, learning_rate=0.01)
    print(round(np.sum(losses), 2))  # 6716.46
