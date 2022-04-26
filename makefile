CC = gcc
CFLAGS = -g -Wall -pthread


server:
	$(CC) $(CFLAGS) server.c threadpool.c syncqueue.c globals.c -o server.out

listTest:
	$(CC) $(CFLAGS) listTest.c list.c -o list.out

hashtableTest:
	$(CC) $(CFLAGS) hashtableTest.c hashtable.c murmur3.c list.c -o hashtable.out

client: client.o
	$(CC) $(CFLAGS) client.c -o client.out
