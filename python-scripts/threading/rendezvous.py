import threading
import time

#global N
N = 5

#global count
count = 0

guard = threading.Semaphore(1)
ready = threading.Semaphore(0)


def rendezvous():
	'''Guard is only allows one thread to increment count at a time
	First N-1 threads to hit ready.acquire() become locked.  The Nth thread
	unlocks one thread which immediately unlocks another, successively letting all pass'''
	global count
	time.sleep(.1)
	guard.acquire()
	count += 1
	if count ==1:
		print('{} string is ready'.format(count))
	else:
		print('{} strings are ready'.format(count))
	guard.release()

	if count == N:
		ready.release()

	ready.acquire()
	ready.release()

	print('Done, the count is {}'.format(count))


thread1 = threading.Thread(target = rendezvous)
thread2 = threading.Thread(target = rendezvous)
thread3 = threading.Thread(target = rendezvous)
thread4 = threading.Thread(target = rendezvous)
thread5 = threading.Thread(target = rendezvous)

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
