#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "SERVER/policy.h"
#include "SERVER/globals.h"
#include "COMMON/macros.h"
#include "COMMON/list.h"

/**
 * @brief Implements a random policy, choosing files in an arbitrary fashion.
 */
long randomHeuristic(void *el)
{
    return rand();
}

/**
 * @brief Implements a fifo policy, choosing the oldest files.
 */
long fifoHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;

    return (long)target->creationTime;
}

/**
 * @brief Implements a lifo policy, choosing the youngest files. It's probably bad.
 */
long lifoHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;

    return -(long)target->creationTime;
}

/**
 * @brief Implements a Least Recently Used policy, choosing the files which were last accessed the most time ago.
 */
long lruHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;

    return (long)target->lastAccess;
}

/**
 * @brief Implements a Most Recently Used policy, choosing the files which were last accessed the least time ago. It's probably bad.
 */
long mruHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;

    return -(long)target->lastAccess;
}

/**
 * @brief Implements a Least Used policy, choosing the files which were used the least in all of the filesystems history.
 */
long luHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;

    return (long)target->numberAccesses;
}

/**
 * @brief Implements a Most Used policy, choosing the files which were used the most in all of the filesystmes history. It's probably bad.
 */
long muHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;

    return -(long)target->numberAccesses;
}

/**
 * @brief Implements a smallest first policy, choosing the smallest files first.
 */
long smolHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;
    return (long)target->size;
}

/**
 * @brief Implements a biggest first policy, choosing the biggest files first.
 */
long biggHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;
    return -(long)target->size;
}

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
    Metadata *target;
    long (*heuristic)(void *);

    switch (POLICY)
    {
    case (P_RAND):
        heuristic = &randomHeuristic;
        break;

    case (P_FIFO):
        heuristic = &fifoHeuristic;
        break;

    case (P_LIFO):
        heuristic = &lifoHeuristic;
        break;

    case (P_LRU):
        heuristic = &lruHeuristic;
        break;

    case (P_MRU):
        heuristic = &mruHeuristic;
        break;

    case (P_MU):
        heuristic = &muHeuristic;
        break;

    case (P_LU):
        heuristic = &luHeuristic;
        break;

    case (P_SMOL):
        heuristic = &smolHeuristic;
        break;

    case (P_BIGG):
        heuristic = &biggHeuristic;
        break;

    default:
        heuristic = &lruHeuristic;
    }

    listSort(fs->filesList, heuristic);

    for (i = 0; i < fs->filesList->size; i++)
    {
        SAFE_ERROR_CHECK(listGet(i, (void **)&target, fs->filesList));
        SAFE_ERROR_CHECK(openFile(target->name, 0, fd, fs));
        if (tryLockFile(*fd, fs) == 0)
        {
            return 0;
        }
        SAFE_ERROR_CHECK(closeFile(*fd, fs));
    }

    errno = ENOENT;
    return -1;
}
