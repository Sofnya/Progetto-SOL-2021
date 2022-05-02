#include <assert.h>
#include <string.h>


#include "../files.h"
#include "../COMMON/macros.h"


int main(int argc, char const *argv[])
{
    File *file1;
    char *alphabet = "abcdefghijklmnopqrstuvwxyz", *tmp;
    void *content;


    puts("Starting tests on File...");
    UNSAFE_NULL_CHECK(file1 = malloc(sizeof(File)));
    ERROR_CHECK(fileInit("file1", file1));

    puts("File initialized!");
    
    content = alphabet;

    ERROR_CHECK(fileLock(file1));
    ERROR_CHECK(fileWrite(content, 5, file1));
    ERROR_CHECK(fileUnlock(file1));
    ERROR_CHECK(fileAppend(content, 5, file1));

    assert(!strcmp(getFileName(file1), "file1"));
    assert(getFileSize(file1) == 10);

    tmp = malloc(11);
    tmp[10] = '\00';

    ERROR_CHECK(fileRead(tmp, 11, file1));
    puts(tmp);
    assert(!strcmp("abcdeabcde", tmp));
    puts("Read/Write/Append working!");
    
    free(tmp);
    fileDestroy(file1);
    free(file1);
    
    
    puts("All tests successfull!");

    return 0;
}
