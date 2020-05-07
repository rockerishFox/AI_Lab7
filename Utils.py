import csv
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt

def readData(filename,inpC,outpC):
    data = []
    names = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count ==0:
                names = row
            else:
                data.append(row)
            line_count+=1
    inp=[]
    for i in range(len(data)):
        l=[]
        for col in inpC:
            colInd = names.index(col)
            l.append(float(data[i][colInd]))
        inp.append(l)

    selectedOutp = names.index(outpC)
    outp=[float(data[i][selectedOutp]) for i in range(len(data))]

    return inp,outp

def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()

def plotModel(trainInputs, trainOutputs, testInputs=None, testOutputs=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    xs = np.tile(np.arange(61), (61, 1))
    ys = np.tile(np.arange(61), (61, 1)).T
    x, y = np.meshgrid([trainInputs[i][1] for i in range(len(trainInputs))],
                       [trainInputs[i][2] for i in range(len(trainInputs))])
    ax.plot_surface(x, y, trainOutputs)
    ax.scatter([testInputs[i][0] for i in range(len(testInputs))],
               [testInputs[i][1] for i in range(len(testInputs))], testOutputs, 'b-', marker="^", color="g"'')
    ax.set_xlabel('GDP')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness')
    plt.show()
    model = sklearn.linear_model.LinearRegression()


def plot(regression, trainInputs = None, trainDataOutput = None, testDataInput = None, testDataOutput = None, testing = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if trainDataOutput and trainInputs and  not testing:
        ax.scatter([trainInputs[i][1] for i in range(len(trainInputs))],
                   [trainInputs[i][2] for i in range(len(trainInputs))], trainDataOutput, marker='.', color='red')

    if testDataInput and testDataOutput:
        ax.scatter([testDataInput[i][0] for i in range(len(testDataInput))],
                   [testDataInput[i][1] for i in range(len(testDataInput))], testDataOutput, marker='^', color='green')

    ax.set_xlabel("GDP")
    ax.set_ylabel("Freedom")
    ax.set_zlabel("Happiness")

    x = np.arange(min([trainInputs[i][1] for i in range(len(trainInputs))]),
                  max([trainInputs[i][1] for i in range(len(trainInputs))]), 0.01, float)

    y = np.arange(min([trainInputs[i][2] for i in range(len(trainInputs))]),
                  max([trainInputs[i][2] for i in range(len(trainInputs))]), 0.1, float)
    x, y = np.meshgrid(x, y)

    z = [regression.predict([d1, d2]) for d1, d2 in zip(x,y)]
    z = np.array(z)
    Z = z.reshape(x.shape)

    ax.plot_surface(x, y, Z, alpha=0.5)
    plt.show()