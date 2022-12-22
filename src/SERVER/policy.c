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
 * @brief Implements a least frequently used policy, choosing the files with the least accesses/second since their creation.
 */
long lfuHeuristic(void *el)
{
    Metadata *target = (Metadata *)el;
    double lifespan = difftime(time(NULL), target->creationTime);

    // We multiply the number of Accesses by 1000 to get a decently precise integer result.
    // TODO: A better solution would be implementing a listSort that uses a comparison function.
    if (lifespan == 0)
    {
        lifespan = 1;
    }
    return (target->numberAccesses * 1000) / (long)lifespan;
}

/**
 * @brief Opens and locks a File chosen to be removed from the fileserver on a capacity miss.
 *
 * @param fd where the FileDescriptor will be returned.
 * @param fs the FileSystem on which to operate.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int missPolicy(FileDescriptor **fd, FileSystem *fs)
{
    int i;
    Metadata *target;
    long (*heuristic)(void *);

    // First we choose our sorting heuristic based on global POLICY.
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

    case (P_LFU):
        heuristic = &lfuHeuristic;
        break;

    default:
        heuristic = &lruHeuristic;
    }

    // Then sort the filesList accordingly.
    // As this is an insertionSort, best case complexity is O(n). So we don't need to worry about sorting too often.
    listSort(fs->filesList, heuristic);

    // We then find and lock the first unlocked File available.
    for (i = 0; i < fs->filesList->size; i++)
    {
        SAFE_ERROR_CHECK(listGet(i, (void **)&target, fs->filesList));
        SAFE_ERROR_CHECK(openFile(target->name, 0, "ADMIN:MISSPOLICY", fd, fs));
        if (isLockedFile((*fd)->name, fs) != 0)
        {
            return 0;
        }
        SAFE_ERROR_CHECK(closeFile(*fd, fs));
    }

    errno = ENOENT;
    return -1;
}
