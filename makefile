CC = gcc
CFLAGS = -g -Wall -pthread


server:
	$(CC) $(CFLAGS) server.c COMMON/threadpool.c COMMON/syncqueue.c globals.c -o server.out

listTest:
	$(CC) $(CFLAGS) TESTS/listTest.c COMMON/list.c -o list.out

hashtableTest:
	$(CC) $(CFLAGS) TESTS/hashtableTest.c COMMON/hashtable.c COMMON/murmur3.c COMMON/list.c -o hashtable.out

client: client.o
	$(CC) $(CFLAGS) client.c -o client.out

filesTest:
	$(CC) $(CFLAGS) TESTS/filesTest.c files.c files.h COMMON/* -o filesTest.out
