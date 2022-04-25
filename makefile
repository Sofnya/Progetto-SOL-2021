CC = gcc
CFLAGS = -g -Wall -pthread


server:
	$(CC) $(CFLAGS) server.c threadpool.c syncqueue.c globals.c -o server.out

listTest:
	$(CC) $(CFLAGS) listTest.c list.c -o list.out

client: client.o
