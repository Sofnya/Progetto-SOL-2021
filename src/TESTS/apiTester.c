#include <time.h>
#include <stdio.h>


#include "CLIENT/api.h"
#include "COMMON/message.h"

#define SOCKNAME "testAddress"

int main(int argc, char const *argv[])
{
    struct timespec abstime;
    void *buf;
    size_t size;

    timespec_get(&abstime, TIME_UTC);
    abstime.tv_sec += 10;

    openConnection(SOCKNAME, 100, abstime);
    openFile("testFile", O_CREATE);
    lockFile("testFile");


    writeFile("testFile", "./");
    
    appendToFile("testFile", (void *)"aaaa", 5, "./");

    readFile("testFile", &buf, &size);
    readNFiles(10, "./");
    unlockFile("testFile");
    removeFile("testFile");
    closeConnection(SOCKNAME);
    
    puts("all done!");

    return 0;
}
