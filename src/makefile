CC = gcc
CFLAGS = -g -Wall -pthread  -I ../include
LIBRARY = -luuid
obj_dir = ../obj/
client_objects = $(obj_dir)CLIENT/api.o $(obj_dir)CLIENT/client.o 
common_objects = $(obj_dir)COMMON/atomicint.o $(obj_dir)COMMON/fileContainer.o $(obj_dir)COMMON/hashtable.o $(obj_dir)COMMON/helpers.o $(obj_dir)COMMON/list.o $(obj_dir)COMMON/message.o $(obj_dir)COMMON/murmur3.o $(obj_dir)COMMON/syncqueue.o $(obj_dir)COMMON/threadpool.o $(obj_dir)COMMON/logging.o
server_objects = $(obj_dir)SERVER/files.o $(obj_dir)SERVER/filesystem.o $(obj_dir)SERVER/globals.o $(obj_dir)SERVER/server.o $(obj_dir)SERVER/connstate.o 
test_objects = $(obj_dir)TESTS/unitTest.o

objects =  $(client_objects) $(common_objects) $(server_objects) $(test_objects)

.PHONY: all server client unitTest clean stressTester apiTester

all: $(objects)

server: $(common_objects) $(server_objects)
	$(CC) $(CFLAGS) $(common_objects) $(server_objects) -o ../out/server.out $(LIBRARY)

client: $(client_objects) $(common_objects)
	$(CC) $(CFLAGS) $(client_objects) $(common_objects) -o ../out/client.out

unitTest: $(obj_dir)TESTS/unitTest.o $(obj_dir)SERVER/files.o $(obj_dir)SERVER/filesystem.o $(common_objects) 
	$(CC) $(CFLAGS) $(obj_dir)TESTS/unitTest.o $(obj_dir)SERVER/files.o $(obj_dir)SERVER/filesystem.o $(common_objects) -o ../out/unitTest.out

apiTester: $(obj_dir)TESTS/apiTester.o $(obj_dir)CLIENT/api.o $(common_objects)
	$(CC) $(CFLAGS) $(obj_dir)TESTS/apiTester.o $(obj_dir)CLIENT/api.o $(common_objects) -o ../out/apiTester.out

stressTester: 
	cp TESTS/stressTester.sh  ../out/stressTester.sh

$(objects): $(obj_dir)%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(objects)
	rm -f $(wildcard ../out/*.out)
