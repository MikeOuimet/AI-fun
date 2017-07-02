import threading
import time

#global N


#guard = threading.Semaphore(1)
#turnstile1 = threading.Semaphore(0)
#turnstile2 = threading.Semaphore(1)

class Barrier:
	def __init__(self, N):
		self.count = 0
		self.N = N
		self.mutex = threading.Semaphore(1)
		self.turnstile1 = threading.Semaphore(0)
		self.turnstile2 = threading.Semaphore(1)


	def phase1(self):
		self.mutex.acquire()
		self.count += 1
		#print('Threads ready: {}'.format(self.count))
		if self.count == self.N:
			self.turnstile2.acquire()
			#self.turnstile1.release()
			for sigs in range(self.N+1):
				self.turnstile1.release()
		self.mutex.release()

		self.turnstile1.acquire()
		#self.turnstile1.release()


	def phase2(self):
		self.mutex.acquire()
		self.count -= 1
		#print('Threads through: {}'.format(self.N - self.count))
		if self.count == 0:
			self.turnstile1.acquire()
			#self.turnstile2.release()
			for sigs in range(self.N+1):
				self.turnstile2.release()
		#	print('No more shall pass')
		self.mutex.release()

		self.turnstile2.acquire()
		#self.turnstile2.release()



	def wait(self):
		self.phase1()
		self.phase2()


if __name__ =='__main__':

	N = 5
	b = Barrier(N)

	def rendezvous():
		'''Guard only allows one thread to increment count at a time
		First N-1 threads to hit ready.acquire() become locked.  The Nth thread
		unlocks one thread which immediately unlocks another, successively letting all pass'''
		for i in range(5):
			b.wait()
			print('Done')
			#or
			#b.phase1()
			#time-critical code
			#b.phase2()
			



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
