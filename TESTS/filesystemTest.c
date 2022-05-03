#include <stdlib.h>
#include <assert.h>
#include <string.h>


#include "../filesystem.h"
#include "../COMMON/macros.h"


int main(int argc, char const *argv[])
{
    FileSystem *fs;
    FileDescriptor *fd;
    char *contents = "abcdefghijklmnopqrstuvwxyz", *tmp;

    UNSAFE_NULL_CHECK(fs = malloc(sizeof(FileSystem)));

    ERROR_CHECK(fsInit(100, 2048, fs));

    ERROR_CHECK(openFile("file1", O_CREATE, &fd, fs));
    ERROR_CHECK(writeFile(fd, (void *)contents, 5, fs));
    ERROR_CHECK(appendToFile(fd, (void *)contents, 5, fs));

    UNSAFE_NULL_CHECK(tmp = malloc(11));
    tmp[10] = '\00';

    ERROR_CHECK(readFile(fd, (void **)&tmp, 10, fs));
    puts(tmp);
    assert(!strcmp(tmp, "abcdeabcde"));

    ERROR_CHECK(closeFile(fd, fs));
    free(tmp);
    fsDestroy(fs);
    free(fs);

    puts("All tests succesfull!");
    return 0;
}
