import time
import threading
import matplotlib.pyplot as plt



global data
data = []


def dataInput():
    start = time.time()
    while True:
        time.sleep(.001)
        data.append(time.time() - start)


def plotter():
    while True:
        start = time.time()
        pd = data.copy()  #had to copy data because data changing mid-scatter call, #could also just note index
        plt.scatter(pd, range(len(pd)))
        plt.pause(.1)
        print(time.time() - start)
        print('')



thread1 = threading.Thread(target = dataInput)
thread2 = threading.Thread(target = plotter)

thread1.start()
thread2.start()
