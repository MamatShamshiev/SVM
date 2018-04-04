import numpy as np
import cvxopt


class SVMSolver:
    """
    Класс с реализацией SVM через метод внутренней точки.
    """

    def __init__(self, C, method, kernel='linear', degree=2, gamma=1):
        """
        C - float, коэффициент регуляризации

        method - строка, задающая решаемую задачу, может принимать значения:
            'primal' - соответствует прямой задаче
            'dual' - соответствует двойственной задаче
        kernel - строка, задающая ядро при решении двойственной задачи
            'linear' - линейное
            'polynomial' - полиномиальное
            'rbf' - rbf-ядро
        Обратите внимание, что часть функций класса используется при одном методе решения,
        а часть при другом
        """

        self.C = C
        self.method = method
        self.kernel = kernel
        self.d = degree
        self.gamma = gamma
        
    def get_params(self, deep=False):
        return {'C': self.C, 'method': self.method, 'kernel': self.kernel, 
                'degree': self.d, 'gamma': self.gamma}    

    def compute_primal_objective(self, X, y):
        """
        Метод для подсчета целевой функции SVM для прямой задачи

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """

        M = 1 - y * (X.dot(self.w) + self.w_0)
        M[M < 0] = 0
        return (self.C / X.shape[0]) * M.sum() + 0.5 * self.w.dot(self.w.T)

    def compute_dual_objective(self, X, y):
        """
        Метод для подсчёта целевой функции SVM для двойственной задачи

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        if self.kernel == 'linear':
            Gramm = X.dot(X.T)
        elif self.kernel == 'polynomial':
            Gramm = self.poly_kernel(X, self.d)
        elif self.kernel == 'rbf':
            Gramm = self.rbf_kernel(X, self.gamma)
        return 0.5 * Gramm.dot(self.lambdas * y).dot(self.lambdas * y) - self.lambdas.sum()

    def fit(self, X, y, tolerance=1e-7, max_iter=100, almost_zero=1e-7):
        """
        Метод для обучения svm согласно выбранной в method задаче

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        tolerance - требуемая точность для метода обучения
        max_iter - максимальное число итераций в методе

        """
        cvxopt.solvers.options['maxiters'] = max_iter
        cvxopt.solvers.options['abstol'] = tolerance
        cvxopt.solvers.options['show_progress'] = False
        if self.method == 'primal':
            diag = np.hstack((0, np.ones(X.shape[1]), np.zeros(X.shape[0])))
            P = cvxopt.matrix(np.diag(diag), tc='d')
            q = cvxopt.matrix(np.hstack((np.zeros(X.shape[1] + 1), 
                                         np.full(X.shape[0], self.C / X.shape[0]))), tc='d')
            G_high = np.hstack((-y[:, np.newaxis], -y[:, np.newaxis] * X, -np.identity(X.shape[0])))
            G_low = np.hstack((np.zeros((X.shape[0], X.shape[1] + 1)), -np.identity(X.shape[0])))
            G = cvxopt.matrix(np.vstack((G_high, G_low)), tc='d')
            h = cvxopt.matrix(np.hstack((-np.ones(X.shape[0]), np.zeros(X.shape[0]))), tc='d')
            sol = cvxopt.solvers.qp(P, q, G, h)
            self.w_0 = np.array(sol['x'][0])
            self.w = np.array(sol['x'][1:X.shape[1] + 1]).ravel()
            self.ksi = np.array(sol['x'][X.shape[1] + 1:]).ravel()
        elif self.method == 'dual':
            if self.kernel == 'linear':
                Gramm = X.dot(X.T)
            elif self.kernel == 'polynomial':
                Gramm = self.poly_kernel(X, self.d)
            elif self.kernel == 'rbf':
                Gramm = self.rbf_kernel(X, self.gamma)
            else:
                raise TypeError("SVMSolver.fit: kernel must be 'linear', 'polynomial' or 'rbf'!")
            Y = y[np.newaxis, :] * y[:, np.newaxis]
            P = cvxopt.matrix(Gramm * Y, tc='d')
            q = cvxopt.matrix(np.full(X.shape[0], -1), tc='d')
            G = cvxopt.matrix(np.vstack((np.identity(X.shape[0]), 
                                         (-1) * np.identity(X.shape[0]))), tc='d')
            h = cvxopt.matrix(np.hstack((np.full(X.shape[0], self.C / X.shape[0]), 
                                         np.zeros(X.shape[0]))), tc='d')
            A = cvxopt.matrix(y[np.newaxis, :], tc='d')
            b = cvxopt.matrix(0, tc='d')
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            self.lambdas = np.array(sol['x']).reshape(X.shape[0])
            self.lambdas[np.isclose(self.lambdas, 0, atol=almost_zero, rtol=0)] = 0
            self.sv = X[self.lambdas > 0]
            self.sv_y = y[self.lambdas > 0]
            self.w_0 = (1 / len(self.sv) * 
                        (y[self.lambdas > 0].sum() - 
                         (self.lambdas * y * Gramm)[[self.lambdas > 0]][:, self.lambdas > 0].sum()))
        else:
            raise TypeError("SVMSolver.fit: method must be 'primal' or 'dual'!")

    def predict(self, X):
        """
        Метод для получения предсказаний на данных

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        """

        if self.method == 'primal':
            return np.where(X.dot(self.w) + self.w_0 > 0, 1, -1)
        elif self.method == 'dual':
            if self.kernel == 'linear':
                Gramm = X.dot(self.sv.T)
            elif self.kernel == 'polynomial':
                Gramm = self.poly_kernel(X1=X, X2=self.sv, d=self.d)
            elif self.kernel == 'rbf':
                Gramm = self.rbf_kernel(X1=X, X2=self.sv, gamma=self.gamma)
            return np.where((self.lambdas[self.lambdas > 0] * self.sv_y * Gramm).sum(axis=1) +
                            self.w_0 > 0, 1, -1)
            
    def get_w(self, X=None, y=None):
        """
        Получить прямые переменные (без учёта w_0)

        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y)

        return: одномерный numpy array
        """
        if self.method == 'primal':
            return self.w
        elif self.method == 'dual':
            return ((self.lambdas[self.lambdas > 0] * self.sv_y)[:, np.newaxis] * 
                    self.sv).sum(axis=0)

    def get_w0(self, X=None, y=None):
        """
        Получить вектор сдвига

        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y)

        return: float
        """
        return self.w_0

    def get_dual(self):
        """
        Получить двойственные переменные

        return: одномерный numpy array
        """
        return self.lambdas

    def poly_kernel(self, X1, d, X2=None):
        if X2 is None:
            X2 = X1
        return (1 + X1.dot(X2.T)) ** d

    def rbf_kernel(self, X1, gamma,  X2=None):
        if X2 is None:
            X2 = X1
        norms1 = (X1 ** 2).sum(axis=1)
        norms2 = (X2 ** 2).sum(axis=1)
        return np.exp(-gamma * (norms1[:, np.newaxis] + norms2[np.newaxis, :] - 2 * X1.dot(X2.T)))
