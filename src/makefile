CC = gcc
CFLAGS = -g -Wall -pthread  -I ../include 
LIBRARY = -luuid -lz -L ../deps
obj_dir = ../obj/
client_objects = $(obj_dir)CLIENT/api.o $(obj_dir)CLIENT/client.o $(obj_dir)CLIENT/clientHelpers.o
common_objects = $(obj_dir)COMMON/atomicint.o $(obj_dir)COMMON/fileContainer.o $(obj_dir)COMMON/hashtable.o $(obj_dir)COMMON/helpers.o $(obj_dir)COMMON/list.o $(obj_dir)COMMON/message.o $(obj_dir)COMMON/murmur3.o $(obj_dir)COMMON/syncqueue.o  
server_objects = $(obj_dir)SERVER/files.o $(obj_dir)SERVER/filesystem.o $(obj_dir)SERVER/globals.o $(obj_dir)SERVER/server.o $(obj_dir)SERVER/connstate.o  $(obj_dir)SERVER/policy.o $(obj_dir)SERVER/threadpool.o $(obj_dir)SERVER/logging.o
test_objects = $(obj_dir)TESTS/unitTest.o $(obj_dir)TESTS/apiTester.o
parser_objects = $(obj_dir)PARSER/parser.o

objects =  $(client_objects) $(common_objects) $(server_objects) $(test_objects) $(parser_objects)

.PHONY: all server client unitTest clean stressTester apiTester test1 test2 test3 parser

all: $(objects)

server: $(common_objects) $(server_objects) parser
	$(CC) $(CFLAGS) $(common_objects) $(server_objects) -o ../out/server.out $(LIBRARY)
	cp SERVER/config.txt ../out/config.txt

client: $(client_objects) $(common_objects)
	$(CC) $(CFLAGS) $(client_objects) $(common_objects) -o ../out/client.out $(LIBRARY)

unitTest: $(obj_dir)TESTS/unitTest.o $(obj_dir)SERVER/files.o $(obj_dir)SERVER/filesystem.o $(obj_dir)SERVER/policy.o $(obj_dir)SERVER/logging.o $(obj_dir)SERVER/globals.o $(obj_dir)SERVER/threadpool.o $(common_objects) 
	$(CC) $(CFLAGS) $(obj_dir)TESTS/unitTest.o $(obj_dir)SERVER/files.o $(obj_dir)SERVER/filesystem.o $(obj_dir)SERVER/policy.o $(obj_dir)SERVER/logging.o $(obj_dir)SERVER/globals.o $(obj_dir)SERVER/threadpool.o $(common_objects) -o ../out/unitTest.out $(LIBRARY)

apiTester: $(obj_dir)TESTS/apiTester.o $(obj_dir)CLIENT/api.o $(common_objects)
	$(CC) $(CFLAGS) $(obj_dir)TESTS/apiTester.o $(obj_dir)CLIENT/api.o $(common_objects) -o ../out/apiTester.out $(LIBRARY)

stressTester: 
	cp TESTS/stressTester.sh  ../out/stressTester.sh

parser: $(parser_objects) $(common_objects)
	$(CC) $(CFLAGS) $(parser_objects) $(common_objects) -o ../out/parser.out $(LIBRARY)
	cp PARSER/statistiche.sh ../out/

test1: server client
	cp -r TESTS/test1 ../out/

	cp TESTS/setupTestFiles.sh TESTS/createRandomFiles.sh ../out/test1
	cd ../out/test1; ./setupTestFiles.sh
	rm -f ../out/test1/setupTestFiles.sh ../out/test1/createRandomFiles.sh

	cd ../out/test1; ./test1.sh

test2: server client
	cp -r TESTS/test2 ../out/

	cp TESTS/setupTestFiles.sh TESTS/createRandomFiles.sh ../out/test2
	cd ../out/test2; ./setupTestFiles.sh
	rm -f ../out/test2/setupTestFiles.sh ../out/test2/createRandomFiles.sh

	cd ../out/test2; ./test2.sh

test3: server client
	cp -r TESTS/test3 ../out/

	cp TESTS/setupTestFiles.sh TESTS/createRandomFiles.sh ../out/test3
	cd ../out/test3; ./setupTestFiles.sh
	rm -f ../out/test3/setupTestFiles.sh ../out/test3/createRandomFiles.sh

	cd ../out/test3; ./test3.sh


$(objects): $(obj_dir)%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(objects)
	rm -rf ../out/*
