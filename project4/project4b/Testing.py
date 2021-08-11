from NeuralNetUtil import buildExamplesFromCarData, buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt


def average(argList):
    return sum(argList) / float(len(argList))


def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val - mean), 2) for val in argList]
    return sqrt(sum(diffSq) / len(argList))


penData = buildExamplesFromPenData()


def testPenData(hiddenLayers=[24]):
    return buildNeuralNet(penData, maxItr=200, hiddenLayerList=hiddenLayers)


carData = buildExamplesFromCarData()


def testCarData(hiddenLayers=[16]):
    return buildNeuralNet(carData, maxItr=200, hiddenLayerList=hiddenLayers)


# testCarData()

# Method for question 5
def q5():
    print("Project 4, Question 5\n")

    # Run 5 iterations of testPenData and testCarData
    # with default parameters and report the max, average, and standard deviation of the accuracy
    penDataResults = []
    carDataResults = []

    for i in range(5):
        # Get testAccuracy from the methods
        penDataResults.append(testPenData()[1])
        print("testPenData() has ran " + str(i + 1) + " times\n")
        carDataResults.append(testCarData()[1])
        print("testCarData() has ran " + str(i + 1) + " times\n")

    # Print the results
    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print("Pen data results: ")
    print("Max of Pen Data: " + str(max(penDataResults)))
    print("Average of Pen Data: " + str(average(penDataResults)))
    print("Standard Deviation of Pen Data: " + str(stDeviation(penDataResults)) + "\n")

    print("Car data results: ")
    print("Max of Car Data: " + str(max(carDataResults)))
    print("Average of Car Data: " + str(average(carDataResults)))
    print("Standard Deviation of Car Data: " + str(stDeviation(carDataResults)) + "\n")


# Method for Question 6
def q6():
    print("Project 4, Question 6\n")

    # Vary the amount of perceptrons in thr hidden layer from 0 to 40 inclusive in increments of 5
    # and get the max, average, and standard deviation of 5 runs of testPenData and testCarData
    # for each number of perceptrons.
    penDataOverall = {}
    carDataOverall = {}

    for x in range(0, 41, 5):
        penDataResults = []
        carDataResults = []

        for y in range(5):
            # Get testAccuracy from the methods
            print("Pen test number: " + str(y + 1) + " x: " + str(x))
            penDataResults.append(testPenData([y])[1])

            print("Car test number: " + str(y + 1) + " x: " + str(x))
            carDataResults.append(testCarData([y])[1])

        # Store Results
        penStats = {}
        penStats['max'] = max(penDataResults)
        penStats['average'] = average(penDataResults)
        penStats['stDev'] = stDeviation(penDataResults)
        penDataOverall[x] = (penDataResults, penStats)

        carStats = {}
        carStats['max'] = max(carDataResults)
        carStats['average'] = average(carDataResults)
        carStats['stDev'] = stDeviation(carDataResults)
        carDataOverall[x] = (carDataResults, carStats)

    print("~~~ Q6 ~~~")
    print("\nPEN DATA")
    for p in penDataOverall:
        print("Perceptrons: " + str(p))
        print(penDataOverall[p])

    print("\nCAR DATA")
    for p in carDataOverall:
        print("Perceptrons: " + str(p))
        print(carDataOverall[p])


q5()
#q6()
