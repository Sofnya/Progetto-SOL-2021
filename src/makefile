CC = gcc
CFLAGS = -g -Wall -pthread  -I ../include
LIBRARY = -luuid
client_objects = CLIENT/api.o CLIENT/client.o 
common_objects = COMMON/atomicint.o COMMON/fileContainer.o COMMON/hashtable.o COMMON/helpers.o COMMON/list.o COMMON/message.o COMMON/murmur3.o COMMON/syncqueue.o COMMON/threadpool.o 
server_objects = SERVER/files.o SERVER/filesystem.o SERVER/globals.o SERVER/server.o SERVER/connstate.o
test_objects = TESTS/unitTest.o
objects =  $(client_objects) $(common_objects) $(server_objects) $(test_objects)

.PHONY: all server client unitTest clean

all: $(objects)

server: $(common_objects) $(server_objects)
	$(CC) $(CFLAGS) $(common_objects) $(server_objects) -o ../out/server.out $(LIBRARY)

client: $(client_objects) $(common_objects)
	$(CC) $(CFLAGS) $(client_objects) $(common_objects) -o ../out/client.out

unitTest: TESTS/unitTest.o SERVER/files.o SERVER/filesystem.o $(common_objects) 
	$(CC) $(CFLAGS) TESTS/unitTest.o SERVER/files.o SERVER/filesystem.o $(common_objects) -o ../out/unitTest.out

apiTester: TESTS/apiTester.o CLIENT/api.o $(common_objects)
	$(CC) $(CFLAGS) TESTS/apiTester.o CLIENT/api.o $(common_objects) -o ../out/apiTester.out

stressTester: 
	cp TESTS/stressTester.sh  ../out/stressTester.sh

$(objects): %.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(objects)
	rm -f $(wildcard ../out/*.out)
