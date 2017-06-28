import time
import threading
import matplotlib.pyplot as plt
import math



global data
data = []


def dataInput():
    start = time.time()
    while True:
        time.sleep(.03)
        data.append(math.sin(time.time() - start)* (time.time() - start))


def plotter():
    while True:
        start = time.time()
        length = len(data)
        plt.scatter(range(length), data[0:length])
        plt.pause(.1)
        print(time.time() - start)
        print('')



thread1 = threading.Thread(target = dataInput)
thread2 = threading.Thread(target = plotter)

thread1.start()
thread2.start()
