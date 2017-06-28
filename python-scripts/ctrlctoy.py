import time


def main():
    try:
        start = time.time()
        print(time.time()-start)
        while(True):
            time.sleep(1)
            print(time.time()-start)

    except KeyboardInterrupt:
        '''Here we close down main'''
        print('Closing main')
        #raise  #if raise, enters except in outer loop


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt')
    finally:
        print("All done")


