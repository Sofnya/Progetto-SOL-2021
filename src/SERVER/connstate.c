#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>

#include "SERVER/connstate.h"
#include "SERVER/filesystem.h"
#include "COMMON/hashtable.h"
#include "COMMON/macros.h"
#include "COMMON/helpers.h"

/**
 * @brief Initializes given ConnState with given FileSystem.
 *
 * @param fs the FileSystem to which the ConnState will refer.
 * @param state the ConnState to initialize.
 * @return int 0 on success, -1 on failure.
 */
int connStateInit(FileSystem *fs, ConnState *state)
{
    SAFE_NULL_CHECK(state->fds = malloc(sizeof(HashTable)));
    state->fs = fs;

    // Every ConnState has it's own UUID.
    genUUID(state->uuid);

    state->lockedFile = NULL;

    PTHREAD_CHECK(pthread_mutex_init(&state->mtx, NULL));
    state->shouldDestroy = 0;
    atomicInit(&state->inUse);
    atomicInit(&state->requestN);
    atomicInit(&state->parsedN);

    // We don't need a huge hashTable for every connection. Assuming clients tend to open less than 8 files at a time seems reasonable.
    return hashTableInit(16, state->fds);
}

/**
 * @brief Destroys the given ConnState, freeing it's resources.
 *
 * @param state the ConnState to destroy.
 */
void connStateDestroy(ConnState *state)
{
    FileDescriptor *fd;

    while (hashTablePop(NULL, (void **)&fd, *(state->fds)) != -1)
    {
        // Gotta make sure we unlock any files that were locked.
        if (fd->flags & FI_LOCK)
        {
            unlockFile(fd, state->fs);
        }

        fdDestroy(fd);
        free(fd);
    }

    pthread_mutex_destroy(&state->mtx);
    atomicDestroy(&state->inUse);
    atomicDestroy(&state->requestN);
    atomicDestroy(&state->parsedN);
    hashTableDestroy(state->fds);
    free(state->fds);
}

/**
 * @brief Opens a File with name path and chosen flags inside the ConnState. Returns FileContainer array fcs of size fcsSize on a capacity miss.
 *
 * @param path the name of the File to be opened.
 * @param flags can be O_LOCK, O_CREATE, 0 or O_LOCK | O_CREATE.
 * @param fcs where an eventual FileContainer will be returned.
 * @param fcsSize where the size of fcs, if any, will be returned.
 * @param state the ConnState in which to open the File.
 * @return int 0 on success, -1 and sets errno to EOVERFLOW on a capacity miss, -1 and sets errno on error.
 */
int conn_openFile(char *path, int flags, FileContainer **fcs, int *fcsSize, ConnState *state)
{
    FileDescriptor *fd, *tmp;
    int capMiss = 0;

    // If we are locking a File, we have to unlock our previous lockedFile, if present, according to the one-file policy.
    if (flags & O_LOCK)
    {
        if (state->lockedFile != NULL)
        {
            unlockFile(state->lockedFile, state->fs);
            state->lockedFile = NULL;
        }
    }

    // Check if we already opened this file.
    if (hashTableGet(path, (void **)&tmp, *state->fds) != -1)
    {
        errno = EINVAL;
        return -1;
    }

    // We have to try to open the file in a while since a freeSpace doesn't always free enough space in a single call.
    errno = 0;
    while (openFile(path, flags, &fd, state->fs, state->uuid) == -1)
    {
        if (errno != EOVERFLOW)
        {
            return -1;
        }

        capMiss = 1;
        *fcsSize = freeSpace(0, fcs, state->fs);
        if (*fcsSize == -1)
        {
            perror("Error on freeSpace");
            return -1;
        }
    }

    // Update our lockedFile
    if (flags & O_LOCK)
    {
        state->lockedFile = fd;
    }

    // On a capacity miss, we have to return -1 and set errno to EOVERFLOW
    if (capMiss)
    {
        hashTablePut((char *)fd->name, fd, *state->fds);

        errno = EOVERFLOW;
        return -1;
    }
    return hashTablePut((char *)fd->name, fd, *state->fds);
}

/**
 * @brief Closes File of name path.
 *
 * @param path the name of the File to close.
 * @param state inside which ConnState to close the file.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int conn_closeFile(const char *path, ConnState *state)
{
    FileDescriptor *fd;

    // If we are closing our currently locked File, the FileSystem will unlock it for us.
    if (state->lockedFile != NULL)
    {
        if (!strcmp(path, state->lockedFile->name))
        {
            state->lockedFile = NULL;
        }
    }

    // Check if the File is open.
    if (hashTableRemove(path, (void **)&fd, *state->fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return closeFile(fd, state->fs);
}

/**
 * @brief Reads already open File of name path inside of given buffer buf.
 *
 * @param path the name of the File to read.
 * @param buf where up to size bytes of the file's content's will be stored.
 * @param size the number of bytes to read inside of buf.
 * @param state the ConnState inside of which we wish to read the File.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int conn_readFile(const char *path, void **buf, size_t size, ConnState *state)
{
    FileDescriptor *fd;

    if (hashTableGet(path, (void **)&fd, *state->fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    return readFile(fd, buf, size, state->fs);
}

/**
 * @brief Writes size bytes from buf to open File of name Path. Can capacity miss, returning errno=EOVERFLOw and putting FileContainers inside of fcs.
 *
 * @param path the name of the File to write.
 * @param buf where the File's contents will be read.
 * @param size how many bytes should be read from buf.
 * @param fcs on a capacity miss, where the buffer of FileContainers will be returned.
 * @param fcsSize on a capacity miss, where the size of fcs will be returned.
 * @param state the ConnState inside of which we wish to write the File.
 * @return int 0 on success, -1 and sets errno=EOVERFLOW on a capacity miss, -1 and sets errno on failure.
 */
int conn_writeFile(const char *path, void *buf, size_t size, FileContainer **fcs, int *fcsSize, ConnState *state)
{
    FileDescriptor *fd;
    int capMiss = 0;

    if (size > state->fs->maxSize)
    {
        errno = EINVAL;
        return -1;
    }

    // First get the FileDescriptor from the ConnState.
    if (hashTableGet(path, (void **)&fd, *state->fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    // Have to write in a while since we need to try again on a capacity miss.
    errno = 0;
    while (writeFile(fd, buf, size, state->fs) == -1)
    {
        if (errno != EOVERFLOW)
        {
            return -1;
        }

        capMiss = 1;

        *fcsSize = freeSpace(size, fcs, state->fs);
        if (*fcsSize == -1)
        {
            perror("Couldn't free space for capacity miss...");
            return -1;
        }
        errno = 0;
    }

    if (capMiss)
    {
        errno = EOVERFLOW;
        return -1;
    }
    return 0;
}

/**
 * @brief Appends size bytes from buf to open File of name Path. Can capacity miss, returning errno=EOVERFLOw and putting FileContainers inside of fcs.
 *
 * @param path the name of the File to append to.
 * @param buf where the File's contents will be read.
 * @param size how many bytes should be read from buf.
 * @param fcs on a capacity miss, where the buffer of FileContainers will be returned.
 * @param fcsSize on a capacity miss, where the size of fcs will be returned.
 * @param state the ConnState inside of which we wish to append to the File.
 * @return int 0 on success, -1 and sets errno=EOVERFLOW on a capacity miss, -1 and sets errno on failure.
 */
int conn_appendFile(const char *path, void *buf, size_t size, FileContainer **fcs, int *fcsSize, ConnState *state)
{
    FileDescriptor *fd;
    int capMiss = 0;

    if (size + getSize(path, state->fs) > state->fs->maxSize)
    {
        errno = EINVAL;
        return -1;
    }

    if (hashTableGet(path, (void **)&fd, *state->fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    errno = 0;
    while (appendToFile(fd, buf, size, state->fs) == -1)
    {
        if (errno != EOVERFLOW)
        {
            return -1;
        }
        capMiss = 1;

        *fcsSize = freeSpace(size, fcs, state->fs);
        if (*fcsSize == -1)
        {
            perror("Couldn't free space for capacity miss...");
            return -1;
        }
        errno = 0;
    }

    if (capMiss)
    {
        errno = EOVERFLOW;
        return -1;
    }

    return 0;
}

/**
 * @brief Reads up to N random Files from given ConnState. Returns them in an array of FileContainers.
 *
 * @param N how many Files we wish to read. Depending on the FileSystem less may be read.
 * @param fcs where the FileContainers will be returned.
 * @param state the ConnState in which we wish to readN Files.
 * @return int on success the actual number of Files read, equals to the size of fcs in number of FileContainers. -1 and sets errno on failure
 */
int conn_readNFiles(int N, FileContainer **fcs, ConnState *state)
{
    return readNFiles(N, fcs, state->uuid, state->fs);
}

/**
 * @brief Remove open File of name path.
 *
 * @param path the name of the File to remove.
 * @param state the ConnState in which we wish to remove the File.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int conn_removeFile(const char *path, ConnState *state)
{
    FileDescriptor *fd;

    if (hashTableRemove(path, (void **)&fd, *state->fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    // If we are removing our lockedFile we need to update it accordingly.
    if (state->lockedFile != NULL)
    {
        if (!strcmp(path, state->lockedFile->name))
        {
            state->lockedFile = NULL;
        }
    }

    if (removeFile(fd, state->fs) != 0)
    {
        return -1;
    }

    fdDestroy(fd);
    free(fd);

    return 0;
}

/**
 * @brief Lock open File of name path.
 *
 * @param path the name of the File to lock.
 * @param state the ConnState in which we wish to lock the File.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int conn_lockFile(const char *path, ConnState *state)
{
    FileDescriptor *fd;

    // If we are locking a new File, automatically unlock the previously locked File according to the one-file policy.
    if (state->lockedFile != NULL)
    {
        if (unlockFile(state->lockedFile, state->fs) != 0)
        {
            errno = EINVAL;
            return -1;
        }
        state->lockedFile = NULL;
    }

    if (hashTableGet(path, (void **)&fd, *state->fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    // And update our lockedFile to the newly locked File.
    state->lockedFile = fd;

    return lockFile(fd, state->fs);
}

/**
 * @brief Unlock open File of name path.
 *
 * @param path the name of the File to unlock.
 * @param state the ConnState in which we wish to unlock the File.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int conn_unlockFile(const char *path, ConnState *state)
{
    FileDescriptor *fd;
    int tmp;

    if (hashTableGet(path, (void **)&fd, *state->fds) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    if ((tmp = unlockFile(fd, state->fs)) == 0)
    {
        // Remember to update our lockedFile.
        state->lockedFile = NULL;
    }
    return tmp;
}
