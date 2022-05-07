CC = gcc
CFLAGS = -g -Wall -pthread


server:
	$(CC) $(CFLAGS) SERVER/*.c COMMON/*.c  globals.c -o ARTIFACTS/server.out

client:
	$(CC) $(CFLAGS) CLIENT/*.c COMMON/*.c -o ARTIFACTS/client.out

filesTest:
	$(CC) $(CFLAGS) TESTS/filesTest.c SERVER/files.c COMMON/* -o ARTIFACTS/filesTest.out

filesystemTest:
	$(CC) $(CFLAGS) TESTS/filesystemTest.c SERVER/filesystem.c SERVER/files.c COMMON/* -o ARTIFACTS/filesystemTest.out

messageTest:
	$(CC) $(CFLAGS) TESTS/messageTest.c COMMON/message.c -o ARTIFACTS/messageTest.out

listTest:
	$(CC) $(CFLAGS) TESTS/listTest.c COMMON/list.c -o ARTIFACTS/list.out

hashtableTest:
	$(CC) $(CFLAGS) TESTS/hashtableTest.c COMMON/hashtable.c COMMON/murmur3.c COMMON/list.c -o ARTIFACTS/hashtable.out

syncqueueTest:
	$(CC) $(CFLAGS) TESTS/syncqueueTest.c COMMON/syncqueue.c -o ARTIFACTS/syncqueueTest.out

threadpoolTest:
	$(CC) $(CFLAGS) TESTS/threadpoolTest.c COMMON/threadpool.c COMMON/syncqueue.c COMMON/list.c -o ARTIFACTS/threadpoolTest.out