#include <time.h>
#include <stdio.h>
#include <pthread.h>


#include "CLIENT/api.h"
#include "COMMON/message.h"
#include "COMMON/macros.h"


#define SOCKNAME "testAddress"


void *test(void *ignore);


int main(int argc, char const *argv[])
{
    int i;
    pthread_t pid;
    for(i = 0; i < 2; i++)
    {
        pthread_create(&pid, NULL, &test, NULL);
        pthread_detach(pid);
    }

    sleep(10);
    return 0;
}


void *test(void *ignore)
{
    puts("hallo!");
    struct timespec abstime;
    void *buf;
    size_t size;

    timespec_get(&abstime, TIME_UTC);
    abstime.tv_sec += 10;

    puts("Opening connection");
    SAFE_ERROR_CHECK(openConnection(SOCKNAME, 100, abstime));
    puts("Opening file");
    SAFE_ERROR_CHECK(openFile("testFile", 0));
    puts("Locking file");
    SAFE_ERROR_CHECK(lockFile("testFile"));
    puts("Reading file");
    SAFE_ERROR_CHECK(readFile("testFile", &buf, &size));
    puts("ReadN files");
    SAFE_ERROR_CHECK(readNFiles(10, "./"));
    puts("Writing file");
    SAFE_ERROR_CHECK(writeFile("testFile", "./"));
    puts("Appending to file");
    SAFE_ERROR_CHECK(appendToFile("testFile", (void *)"aaaa", 5, "./"));
    puts("Unlocking file");
    SAFE_ERROR_CHECK(unlockFile("testFile"));
    puts("Removing file");
    SAFE_ERROR_CHECK(removeFile("testFile"));
    puts("Closing connection");
    SAFE_ERROR_CHECK(closeConnection(SOCKNAME));
    
    puts("all done!");

    return 0;
}