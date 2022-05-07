CC = gcc
CFLAGS = -g -Wall -pthread -I include


server:
	$(CC) $(CFLAGS) src/SERVER/* src/COMMON/* -o ARTIFACTS/server.out

client:
	$(CC) $(CFLAGS) src/CLIENT/* src/COMMON/* -o ARTIFACTS/client.out

filesTest:
	$(CC) $(CFLAGS) src/TESTS/filesTest.c src/SERVER/files.c -o ARTIFACTS/filesTest.out

filesystemTest:
	$(CC) $(CFLAGS) src/TESTS/filesystemTest.c src/SERVER/filesystem.c src/SERVER/files.c src/COMMON/*.c -o ARTIFACTS/filesystemTest.out

messageTest:
	$(CC) $(CFLAGS) src/TESTS/messageTest.c src/COMMON/message.c -o ARTIFACTS/messageTest.out

listTest:
	$(CC) $(CFLAGS) src/TESTS/listTest.c src/COMMON/list.c -o ARTIFACTS/listTest.out

hashtableTest:
	$(CC) $(CFLAGS) src/TESTS/hashtableTest.c src/COMMON/hashtable.c src/COMMON/murmur3.c src/COMMON/list.c -o ARTIFACTS/hashtable.out

syncqueueTest:
	$(CC) $(CFLAGS) src/TESTS/syncqueueTest.c src/COMMON/syncqueue.c -o ARTIFACTS/syncqueueTest.out

threadpoolTest:
	$(CC) $(CFLAGS) src/TESTS/threadpoolTest.c src/COMMON/threadpool.c src/COMMON/syncqueue.c src/COMMON/list.c -o ARTIFACTS/threadpoolTest.out