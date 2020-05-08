import numpy as np


class LS:
    def __init__(self):
        self.w = []  # reprezentare : [ b0 , b1 , b2 ]

    """
    :return: Inmultirea matricei m1 cu m2 ( M1 X M2 )
    """

    def __multiply(self, m1, m2):
        mat = [[0 for i in range(len(m2[0]))] for j in range(len(m1))]
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for x in range(len(m1[0])):
                    mat[i][j] += m1[i][x] * m2[x][j]
        return mat

    """
    :return: Transpusa matricei M ( A ^ T )
    """

    def __transp(self, matrix):
        res = []
        for i in range(len(matrix[0])):
            line = []
            for j in range(len(matrix)):
                line.append(matrix[j][i])
            res.append(line)
        return res


    """
    :return: Matricea fara linia x si coloana y
    """

    def __remove(self, mat, x, y):
        res = []
        for i in range(len(mat)):
            line = []
            for j in range(len(mat[0])):
                if i != x and j != y:
                    line.append(mat[i][j])
            if line:
                res.append(line)
        return res


    """
    :return: Matricea fara prima linie si coloana x 
    """

    def __subMat(self, matrix, x):
        res = []
        for i in range(1, len(matrix)):
            line = []
            for j in range(len(matrix)):
                if j != x:
                    line.append(matrix[i][j])
            res.append(line)
        return res

    """
    :return: Matricea adjuncta a matricei M  ->  A*
    """

    def __adj(self, mat):
        res = []
        for i in range(len(mat)):
            line = []
            for j in range(len(mat[0])):
                x = self.__det(self.__remove(mat, i, j)) * (-1) ** (i + j)
                line.append(x)
            res.append(line)
        return self.__transp(res)

    """
    :return: Determinantul matricei M ( DELTA (M) )
    """

    def __det(self, matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        res = 0
        for i in range(len(matrix)):
            submat = self.__subMat(matrix, i)
            res += ((-1) ** (i + 2)) * matrix[0][i] * self.__det(submat)
        return res

    """
    :return: Inversa matricei M ->  A ^ (-1)

    FORMULA:  A^(-1) = 1 / det(A) * A*
    """
    def __invers(self, mat):
        det = self.__det(mat)
        adj = self.__adj(mat)
        return [[adj[i][j] / det for j in range(len(adj[0]))] for i in range(len(mat))]


    def fit_manual(self, input, output):
        mat = self.__multiply(self.__transp(input), input)
        mat2 = self.__invers(mat)
        mat3 = self.__multiply(mat2, self.__transp(input))
        w = self.__multiply(mat3, output)
        self.w = w
        return w


    """
    w = ( X ^ t  *  X )^(-1) * X ^ t * Y   
    """
    def fit_tool(self, input, output):

        trans = np.array(input).transpose()
        mul = np.matmul(trans, np.array(input))
        invers = np.linalg.inv(mul)
        mat = np.matmul(invers, trans)
        w = np.matmul(mat, np.array(output))

        self.w = w
        return w

    """
    data = input data [ factor1 , factor2 ]
    """

    def predict(self, data):
        # aplicam formula y = b + b1 * x1 + b2 * x2
        y = self.w[0][0]

        for i in range(len(data)):
            y += data[i] * self.w[i + 1][0]
        return y
