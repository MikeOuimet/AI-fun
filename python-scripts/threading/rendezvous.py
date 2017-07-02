import threading


#global N
N = 5
count = 0
guard = threading.Semaphore(1)
turnstile1 = threading.Semaphore(0)
turnstile2 = threading.Semaphore(1)



def rendezvous():
	'''Guard only allows one thread to increment count at a time
	First N-1 threads to hit ready.acquire() become locked.  The Nth thread
	unlocks one thread which immediately unlocks another, successively letting all pass'''
	global count
	for i in range(5):
		#print('Round {}'.format(i))
		#time.sleep(.1)
		guard.acquire()
		count += 1
		if count ==1:
			print('{} string is ready'.format(count))
		else:
			print('{} strings are ready'.format(count))
		if count == N:
			turnstile2.acquire()
			turnstile1.release()
		guard.release()



		turnstile1.acquire()
		turnstile1.release()


		'''Counts the number of threads through and shuts guard after desired N are through'''
		#Here is where timing-sensitive code would go
		guard.acquire()
		count -=1
		if N-count ==1:
			print('{} string through'.format(N-count))
		else:
			print('{} strings through'.format(N-count))
		if count == 0:
			turnstile1.acquire()
			print('No more shall pass this round')
			print('')
			turnstile2.release()
		guard.release()

		turnstile2.acquire()
		turnstile2.release()






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
