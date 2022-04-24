CC = gcc
CFLAGS = -g -Wall -pthread


server:
	$(CC) $(CFLAGS) server.c threadpool.c syncqueue.c -o server.out

client: client.o
