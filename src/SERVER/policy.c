#include <pthread.h>
#include <stdio.h>

#include "SERVER/policy.h"
#include "COMMON/macros.h"
#include "COMMON/list.h"

/**
 * @brief Opens and locks a file chosen to be removed from the fileserver on a capacity miss.
 *
 * @param fd where the file descriptor will be returned.
 * @param fs the FileSystem on which to operate.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int missPolicy(FileDescriptor **fd, FileSystem *fs)
{
    int i;
    char *target;

    puts("Capacity miss...");

    for (i = 0; i < fs->filesList->size; i++)
    {
        printf("Iteration %d\n", i);
        SAFE_ERROR_CHECK(listGet(i, (void **)&target, fs->filesList));
        puts("Get done");
        SAFE_ERROR_CHECK(openFile(target, 0, fd, fs));
        puts("File opened");
        if (tryLockFile(*fd, fs) == 0)
        {
            printf("Target found: %s, returning\n", target);
            return 0;
        }
        puts("Couldn't lock file, continuing");
        SAFE_ERROR_CHECK(closeFile(*fd, fs));
    }

    puts("No target found, returning");

    errno = ENOENT;
    return -1;
}
