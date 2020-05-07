import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Utils import readData, plot
from LS import LS

data_input, data_output = readData("D:\\utils\\faculta\\sem4\\AI\\Laborator\\Lab07\\AI_Lab7\\Data\\2017.csv",
                                   ["Economy..GDP.per.Capita.", "Freedom"], "Happiness.Score")


def main():
    lines = [i for i in range(len(data_input))]

    # choosing de training ones and the test ones
    train_index = np.random.choice(lines, int(0.8 * len(data_input)), replace=False)
    test_index = [i for i in lines if i not in train_index]


    train_inputs = [([1] + data_input[i]) for i in train_index]
    train_outputs = [[data_output[i]] for i in train_index]
    test_inputs = [data_input[i] for i in test_index]
    test_outputs = [data_output[i] for i in test_index]

    regressor = LS()
    print("Ce algoritm de regresie?")
    print("1 - Manual")
    print("2 - With Tool")
    ans = input(">")

    if ans == "1":
        regressor.fit_manual(train_inputs, train_outputs)
    elif ans == "2":
        regressor.fit_tool(train_inputs, train_outputs)

    print("Modelul de regresie: \n"
          " w0 = " + str(regressor.w[0]) + "\n"
                                           " w1 = " + str(regressor.w[1]) + "\n"
                                                                            " w2 = " + str(regressor.w[2]) + "\n")

    computed_output = [regressor.predict(data) for data in test_inputs]

    # generating graphs for the training session, test session, and both
    plot(regressor, train_inputs, train_outputs)
    plot(regressor, train_inputs, train_outputs, test_inputs, computed_output)
    plot(regressor, train_inputs, train_outputs, test_inputs, computed_output, True)

    # calculating differences between predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computed_output, test_outputs):
        error += (t1 - t2) ** 2
    error = error / len(test_outputs)
    print("prediction error (manual): ", error)

    regressorTool = LinearRegression()
    regressorTool.fit(train_inputs, train_outputs)
    test_inputs = [[1] + test_inputs[i] for i in range(len(test_inputs))]
    computed_output = regressorTool.predict(test_inputs)
    error = mean_squared_error(test_outputs, computed_output)
    print("prediction error (tool): ", error)


main()
