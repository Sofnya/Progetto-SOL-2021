CC = gcc
CFLAGS = -g -Wall -pthread -I include


server:
	$(CC) $(CFLAGS) src/SERVER/* src/COMMON/* -o ARTIFACTS/server.out

client:
	$(CC) $(CFLAGS) src/CLIENT/* src/COMMON/* -o ARTIFACTS/client.out

unitTest:
	$(CC) $(CFLAGS) src/TESTS/unitTest.c src/SERVER/files.c src/SERVER/filesystem.c src/COMMON/* -o ARTIFACTS/unitTest.out
