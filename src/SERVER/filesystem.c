#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

#include "SERVER/filesystem.h"
#include "COMMON/macros.h"
#include "COMMON/helpers.h"
#include "SERVER/policy.h"
#include "SERVER/logging.h"
#include "SERVER/lockhandler.h"
#include "SERVER/globals.h"

#define READLOCK readLock(fs->rwLock, __LINE__, __func__)
#define WRITELOCK writeLock(fs->rwLock, __LINE__, __func__)
#define UNLOCK unlock(fs->rwLock, __LINE__, __func__)
#define LISTLOCK listLock(fs->filesListMtx, __LINE__, __func__)
#define LISTUNLOCK listUnlock(fs->filesListMtx, __LINE__, __func__)

// These functions are hooks used for debugging purposes. They allow us to print every single lock and unlock operation, and where they appear.
int listLock(pthread_mutex_t *lock, int line, const char *func)
{
    int tmp, tmpErrno;
    tmp = pthread_mutex_lock(lock);
    tmpErrno = errno;
#ifdef DEBUG
    printf("Acquired LISTLOCK on line %d in %s\n", line, func);
#endif
    errno = tmpErrno;
    return tmp;
}
int listUnlock(pthread_mutex_t *lock, int line, const char *func)
{
#ifdef DEBUG
    printf("Releasing LISTLOCK on line %d in %s\n", line, func);
#endif
    return pthread_mutex_unlock(lock);
}
int readLock(pthread_rwlock_t *lock, int line, const char *func)
{
    int tmp, tmpErrno;
    tmp = pthread_rwlock_rdlock(lock);
    tmpErrno = errno;
#ifdef DEBUG
    printf("Acquired READLOCK on line %d in %s\n", line, func);
#endif
    errno = tmpErrno;
    return tmp;
}
int writeLock(pthread_rwlock_t *lock, int line, const char *func)
{
    int tmp, tmpErrno;
    tmp = pthread_rwlock_wrlock(lock);
    tmpErrno = errno;
#ifdef DEBUG
    printf("Acquired WRITELOCK on line %d in %s\n", line, func);
#endif
    errno = tmpErrno;
    return tmp;
}
int unlock(pthread_rwlock_t *lock, int line, const char *func)
{
#ifdef DEBUG
    printf("Releasing LOCK on line %d in %s\n", line, func);
#endif
    return pthread_rwlock_unlock(lock);
}

/**
 * @brief Initializes the given FileSystem with max number of Files maxN and maximum memory occupation maxSize.
 *
 * @param maxN the maximum number of Files in the FileSystem, -1 means unlimited.
 * @param maxSize the maximum size the FileSystem should take up, -1 means unlimited.
 * @param fs the FileSystem to initialize.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int fsInit(size_t maxN, size_t maxSize, int isCompressed, FileSystem *fs)
{
    pthread_mutexattr_t attr;
    pthread_rwlockattr_t rwAttr;
    fs->maxN = maxN;
    fs->maxSize = maxSize;

    SAFE_NULL_CHECK(fs->curN = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(fs->curSize = malloc(sizeof(AtomicInt)));

    atomicInit(fs->curN);
    atomicInit(fs->curSize);

    SAFE_NULL_CHECK(fs->filesList = malloc(sizeof(List)));
    SAFE_NULL_CHECK(fs->filesListMtx = malloc(sizeof(pthread_mutex_t)));
    SAFE_NULL_CHECK(fs->rwLock = malloc(sizeof(pthread_rwlock_t)));

    SAFE_NULL_CHECK(fs->filesTable = malloc(sizeof(HashTable)));
    SAFE_NULL_CHECK(fs->lockHandlerQueue = malloc(sizeof(SyncQueue)));

    PTHREAD_CHECK(pthread_mutexattr_init(&attr));
    PTHREAD_CHECK(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE));
    PTHREAD_CHECK(pthread_mutex_init(fs->filesListMtx, &attr));
    PTHREAD_CHECK(pthread_rwlockattr_init(&rwAttr));
    PTHREAD_CHECK(pthread_rwlock_init(fs->rwLock, &rwAttr));
    PTHREAD_CHECK(pthread_mutexattr_destroy(&attr));

    SAFE_ERROR_CHECK(listInit(fs->filesList));
    syncqueueInit(fs->lockHandlerQueue);

    if (maxN > 0)
    {
        // An hashTable of two times the maximum size should give excellent performance and very few collisions.
        SAFE_ERROR_CHECK(hashTableInit(maxN * 2, fs->filesTable));
    }
    else
    {
        // If we don't have this limit we guess a reasonable size.
        SAFE_ERROR_CHECK(hashTableInit(4096, fs->filesTable));
    }

    fs->isCompressed = isCompressed;

    SAFE_NULL_CHECK(fs->fsStats = malloc(sizeof(FSStats)));
    SAFE_ERROR_CHECK(statsInit(fs->fsStats));

    return 0;
}

/**
 * @brief Destroys the given FileSystem, freeing it's resources.
 *
 * @param fs the FileSystem to be destroyed.
 */
void fsDestroy(FileSystem *fs)
{
    Metadata *cur;
    File *curFile;

    logger("Destroying fileSystem", "STATUS");

    // We have to individually destroy all the files still present in the FileSystem.
    while (listSize(*fs->filesList) > 0)
    {
        listPop((void **)&cur, fs->filesList);
        if (hashTableRemove(cur->name, (void **)&curFile, *fs->filesTable) == 0)
        {
            fileDestroy(curFile);
            free(curFile);
        }
    }

    listDestroy(fs->filesList);
    hashTableDestroy(fs->filesTable);
    pthread_mutex_destroy(fs->filesListMtx);
    pthread_rwlock_destroy(fs->rwLock);
    atomicDestroy(fs->curN);
    atomicDestroy(fs->curSize);
    syncqueueDestroy(fs->lockHandlerQueue);

    free(fs->lockHandlerQueue);
    free(fs->filesList);
    free(fs->filesTable);
    free(fs->filesListMtx);
    free(fs->rwLock);
    free(fs->curN);
    free(fs->curSize);

    statsDestroy(fs->fsStats);
    free(fs->fsStats);
}

/**
 * @brief Initializes given FileDescriptor with given flags and name.
 *
 * @param name the name of the new FileDescriptor.
 * @param pid the pid of the calling process.
 * @param flags the flags of the FileDescriptor.
 * @param uuid the UUID of the connection which owns the FileDescriptor.
 * @param fd the FileDescriptor to be initialized.
 * @return int 0 on success, -1 on failure.
 */
int fdInit(const char *name, pid_t pid, int flags, char *uuid, FileDescriptor *fd)
{
    size_t nameLen = strlen(name) + 1;

    fd->pid = pid;
    fd->flags = flags;
    SAFE_NULL_CHECK(fd->name = malloc(nameLen * sizeof(char)));
    SAFE_NULL_CHECK(fd->uuid = malloc(strlen(uuid) + 1));

    strcpy(fd->name, name);
    strcpy(fd->uuid, uuid);

    return 0;
}

/**
 * @brief Destroys given FileDescriptor, freeing it's resources.
 *
 * @param fd the FileDescriptor to be destroyed.
 */
void fdDestroy(FileDescriptor *fd)
{
    free(fd->name);
    free(fd->uuid);
    fd->name = NULL;
}

/**
 * @brief Opens a File, returning a FileDescriptor with given flags.
 *
 * @param pathname the name of the File to open.
 * @param flags can be O_CREATE to create a new file, O_LOCK to lock the file on open, both ORED together or neither.
 * @param uuid the UUID of requesting connection.
 * @param fd where the FileDescriptor will be returned.
 * @param fs the FileSystem containing the chosen File.
 * @return int 0 on success, -1 on failure.
 */
int openFile(char *pathname, int flags, char *uuid, FileDescriptor **fd, FileSystem *fs)
{
    File *file;
    FileDescriptor *newFd;
    HandlerRequest *request;
    char log[500];

    // The only way to create a file in the FileSystem is to go through here.
    if (flags & O_CREATE)
    {
        PTHREAD_CHECK(LISTLOCK);
        PTHREAD_CHECK(WRITELOCK);

        // We make a file with the given name doesn't already exist.
        if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == 0)
        {
            UNLOCK;
            PTHREAD_CHECK(LISTUNLOCK);
            errno = EINVAL;
            return -1;
        }

        // And that creating a new file won't put us over the maximum number of files.
        if (atomicGet(fs->curN) + 1 > fs->maxN)
        {
            UNLOCK;
            LISTUNLOCK;

            errno = EOVERFLOW;
            return -1;
        }

        // Notify the lockHandler of a lock create, which is only needed for the ONE_LOCK_POLICY
        if (ONE_LOCK_POLICY && (flags & O_LOCK) && (flags & O_CREATE))
        {
            UNSAFE_NULL_CHECK(request = malloc(sizeof(HandlerRequest)));
            handlerRequestInit(R_LOCK_CREATE_NOTIFY, pathname, uuid, NULL, request);
            syncqueuePush((void *)request, fs->lockHandlerQueue);
        }

        // If everything is fine, we create the file, initialize it, and put it inside of the FileSystem.
        CLEANUP_CHECK(file = malloc(sizeof(File)), NULL, {
            UNLOCK;
            LISTUNLOCK;
        });

        CLEANUP_ERROR_CHECK(fileInit(pathname, fs->isCompressed, file), {
            UNLOCK;
            LISTUNLOCK;
        });

        CLEANUP_ERROR_CHECK(hashTablePut(pathname, (void *)file, *fs->filesTable), {
            UNLOCK;
            LISTUNLOCK;
        });

        CLEANUP_ERROR_CHECK(listAppend((void *)file->metadata, fs->filesList), {
            UNLOCK;
            LISTUNLOCK;
        });

        // Always update the FileSystems size.
        atomicInc(1, fs->curN);
        atomicInc(getFileTrueSize(file), fs->curSize);

        sprintf(log, ">curN:%ld >curSize:%ld", atomicGet(fs->curN), atomicGet(fs->curSize));
        logger(log, "SIZE");

        statsUpdateSize(fs->fsStats, atomicGet(fs->curSize), atomicGet(fs->curN));
        atomicInc(1, fs->fsStats->filesCreated);

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        fdInit(file->name, getTID(), FI_READ | FI_WRITE | FI_APPEND | FI_CREATED, uuid, newFd);

        // Afterwards we lock the file if needed.
        if (flags & O_LOCK)
        {
            if (fileLock(file, uuid) == -1)
            {
                UNLOCK;
                LISTUNLOCK;
                fdDestroy(newFd);
                free(newFd);
                return -1;
            }
            newFd->flags |= FI_LOCK;
        }
        PTHREAD_CHECK(UNLOCK);
        PTHREAD_CHECK(LISTUNLOCK);
    }
    // On a normal open we just get the file from the filesTable, if it's present.
    else
    {
        PTHREAD_CHECK(READLOCK);

        if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
        {
            UNLOCK;
            errno = EINVAL;
            return -1;
        }
        CLEANUP_CHECK(newFd = malloc(sizeof(FileDescriptor)), NULL, UNLOCK);
        fdInit(file->name, getTID(), FI_READ | FI_APPEND, uuid, newFd);
        if (flags & O_LOCK)
        {
            newFd->flags |= FI_LOCK;
        }
        PTHREAD_CHECK(UNLOCK);
    }

    *fd = newFd;

    atomicInc(1, fs->fsStats->open);
    return 0;
}

/**
 * @brief Closes a File.
 *
 * @param fd a FileDescriptor gotten from opening the chosen File.
 * @param fs the FileSystem containing the chosen file.
 * @return int 0 on success, -1 on failure
 */
int closeFile(FileDescriptor *fd, FileSystem *fs)
{
    HandlerRequest *request;
    if (isLockedByFile(fd->name, fd->uuid, fs) == 0)
    {
        PRINT_ERROR_CHECK(unlockFile(fd->name, fs));

        UNSAFE_NULL_CHECK(request = malloc(sizeof(HandlerRequest)));
        handlerRequestInit(R_UNLOCK_NOTIFY, fd->name, fd->uuid, NULL, request);
        syncqueuePush((void *)request, fs->lockHandlerQueue);
    }

    fdDestroy(fd);
    free(fd);

    atomicInc(1, fs->fsStats->close);

    return 0;
}

/**
 * @brief Reads the contents of the File in fd, and puts them in buf.
 *
 * @param fd a FileDescriptor gotten from opening the chosen File.
 * @param buf the buffer in which the file's contents will be stored.
 * @param size the size of buf, if not large enough to hold all of the file's contents an error will be returned.
 * @param fs the FileSystem containing the chosen file.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int readFile(FileDescriptor *fd, void **buf, size_t size, FileSystem *fs)
{
    File *file;
    int success, tmpErr;

    // Always unset the created flag on any operation with an fd.
    fd->flags &= ~FI_CREATED;

    if (!(fd->flags & FI_READ))
    {
        errno = EINVAL;
        return -1;
    }

    PTHREAD_CHECK(READLOCK);

    if (hashTableGet(fd->name, (void **)&file, *fs->filesTable) == -1)
    {
        UNLOCK;
        errno = EBADF;
        return -1;
    }

    if (fileIsLocked(file) == 0)
    {
        if (fileIsLockedBy(file, fd->uuid) == -1)
        {
            PTHREAD_CHECK(UNLOCK);
            errno = EINVAL;
            return -1;
        }
    }

    if (getFileSize(file) > size)
    {
        PTHREAD_CHECK(UNLOCK);
        errno = ENOBUFS;
        return -1;
    }

    success = fileRead(*buf, size, file);
    tmpErr = errno;

    PTHREAD_CHECK(UNLOCK);

    if (success == 0)
    {
        atomicInc(1, fs->fsStats->read);
    }
    errno = tmpErr;
    return success;
}

/**
 * @brief Writes the contents of buffer buf to the File described by fd.
 *
 * @param fd a FileDescriptor gotten from opening the chosen File.
 * @param buf the buffer from which the contents will be copied.
 * @param size the size of buf, the File will be resized accordingly.
 * @param fs the FileSystem containing the chosen File.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int writeFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs)
{
    File *file;
    size_t oldSize, newSize;
    char log[500];
    if (!((fd->flags & FI_WRITE) && (fd->flags & FI_CREATED) && (isLockedByFile(fd->name, fd->uuid, fs) == 0)))
    {
        errno = EINVAL;
        return -1;
    }
    if (atomicGet(fs->curSize) + size > fs->maxSize)
    {
        errno = EOVERFLOW;
        return -1;
    }

    PTHREAD_CHECK(WRITELOCK);
    CLEANUP_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable), { UNLOCK; });

    oldSize = getFileTrueSize(file);
    CLEANUP_ERROR_CHECK(fileWrite(buf, size, file), { UNLOCK; });
    newSize = getFileTrueSize(file);
    PTHREAD_CHECK(UNLOCK);

    atomicInc(newSize - oldSize, fs->curSize);

    sprintf(log, ">curN:%ld >curSize:%ld", atomicGet(fs->curN), atomicGet(fs->curSize));
    logger(log, "SIZE");

    atomicInc(1, fs->fsStats->write);
    statsUpdateSize(fs->fsStats, atomicGet(fs->curSize), atomicGet(fs->curN));

    // Automatically unsets the FI_CREATED flag, preventing further writes.
    fd->flags &= ~FI_CREATED;

    return 0;
}

/**
 * @brief Appends the contents of buf to the File described by fd.
 *
 * @param fd a FileDescriptor gotten from opening the chosen File.
 * @param buf the buffer from which the contents will be copyed.
 * @param size the size of buf.
 * @param fs the FileSystem containing the chosen File.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int appendToFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs)
{
    File *file;
    size_t oldSize, newSize;
    char log[500];

    // Always unset the created flag on any operation with an fd.
    fd->flags &= ~FI_CREATED;

    if (!(fd->flags & FI_APPEND))
    {
        errno = EINVAL;
        return -1;
    }
    if (atomicGet(fs->curSize) + size > fs->maxSize)
    {
        errno = EOVERFLOW;
        return -1;
    }

    PTHREAD_CHECK(WRITELOCK);
    CLEANUP_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable), { UNLOCK; });

    if (fileIsLocked(file) == 0)
    {
        if (fileIsLockedBy(file, fd->uuid) == -1)
        {
            PTHREAD_CHECK(UNLOCK);
            errno = EINVAL;
            return -1;
        }
    }

    oldSize = getFileTrueSize(file);
    CLEANUP_ERROR_CHECK(fileAppend(buf, size, file), { UNLOCK; });
    newSize = getFileTrueSize(file);
    PTHREAD_CHECK(UNLOCK);

    atomicInc(newSize - oldSize, fs->curSize);

    sprintf(log, ">curN:%ld >curSize:%ld", atomicGet(fs->curN), atomicGet(fs->curSize));
    logger(log, "SIZE");

    atomicInc(1, fs->fsStats->append);
    statsUpdateSize(fs->fsStats, atomicGet(fs->curSize), atomicGet(fs->curN));

    return 0;
}

/**
 * @brief Reads N Files in the given buffer. Buf and size will be malloced, so remember to free them after use.
 *
 * @param N The number of Files to read, if N < 0 reads all of the servers Files.
 * @param buf Where the Files will be stored, will become an array of buffers.
 * @param uuid The UUID of the requesting connection.
 * @param fs The FileSystem to query.
 * @return int on success, the number of Files actually read, on error -1 and sets errno.
 */
int readNFiles(int N, FileContainer **buf, char *uuid, FileSystem *fs)
{
    FileDescriptor *curFD;
    Metadata *cur = NULL;
    int amount, i = 0, curCount = 0;
    size_t curSize;
    void *curBuffer;

    PTHREAD_CHECK(LISTLOCK);

    // We first calculate how many files we will read, initializing a buffer of the right size.
    amount = atomicGet(fs->curN);
    if (N > 0 && N < amount)
        amount = N;
    SAFE_NULL_CHECK(*buf = malloc(sizeof(FileContainer) * amount));

    while (curCount < N && i < atomicGet(fs->curN))
    {
        // We get the files from the filesList.
        CLEANUP_ERROR_CHECK(listGet(i, (void **)&cur, fs->filesList), { LISTUNLOCK; free(*buf); });
        i++;

        curSize = getSize(cur->name, fs);
        CLEANUP_CHECK(curBuffer = malloc(curSize), NULL, { LISTUNLOCK; });

        if (openFile(cur->name, 0, uuid, &curFD, fs) == -1)
        {
            free(curBuffer);
            continue;
        }
        if (readFile(curFD, &curBuffer, curSize, fs) == -1)
        {
            free(curBuffer);
            fdDestroy(curFD);
            free(curFD);
            continue;
        }
        if (closeFile(curFD, fs) == -1)
        {
            free(curBuffer);
            fdDestroy(curFD);
            free(curFD);
            continue;
        }

        // And put them in a container inside of the output buffer.
        CLEANUP_ERROR_CHECK(containerInit(curSize, curBuffer, cur->name, &((*buf)[curCount])), {LISTUNLOCK;  free(*buf); });
        free(curBuffer);
        curCount++;
    }
    *buf = realloc(*buf, sizeof(FileContainer) * curCount);
    PTHREAD_CHECK(LISTUNLOCK);

    atomicInc(1, fs->fsStats->readN);
    return curCount;
}

/**
 * @brief Atomically locks given File, giving ownership to given UUID.
 *
 * @param name the name of the File to lock.
 * @param uuid the chosen owner of the File.
 * @param fs the FileSystem containing given File.
 * @return int 0 on success, -1 on failure.

 */
int lockFile(char *name, char *uuid, FileSystem *fs)
{
    File *file;
    int tmp;

    PTHREAD_CHECK(READLOCK);

    if (hashTableGet(name, (void **)&file, *fs->filesTable) == -1)
    {
        UNLOCK;
        errno = EBADF;
        return -1;
    }
    tmp = fileLock(file, uuid);
    UNLOCK;
    atomicInc(1, fs->fsStats->lock);

    return tmp;
}

/**
 * @brief Unlocks given File.
 *
 * @param name the name of the File to unlock.
 * @param fs the FileSystem containing given File.
 * @return int 0 on success, -1 on failure.
 */
int unlockFile(char *name, FileSystem *fs)
{
    File *file;

    PTHREAD_CHECK(READLOCK);

    CLEANUP_ERROR_CHECK(hashTableGet(name, (void **)&file, *fs->filesTable), UNLOCK);

    CLEANUP_ERROR_CHECK(fileUnlock(file), UNLOCK);
    PTHREAD_CHECK(UNLOCK);

    atomicInc(1, fs->fsStats->unlock);

    return 0;
}

/**
 * @brief Checks whether the File of given name is locked.
 *
 * @param name the name of chosen File.
 * @param fs the FileSystem containing the chosen File.
 * @return int 0 if chosen File is locked, -1 otherwise.
 */
int isLockedFile(char *name, FileSystem *fs)
{
    File *file;

    PTHREAD_CHECK(READLOCK);
    if (hashTableGet(name, (void **)&file, *fs->filesTable) == -1)
    {
        UNLOCK;
        errno = EBADF;
        return -1;
    }
    PTHREAD_CHECK(UNLOCK);

    return fileIsLocked(file);
}

/**
 * @brief Checks whether the File of given name is locked by given UUID.
 *
 * @param name the name of chosen File.
 * @param uuid the UUID to check.
 * @param fs the FileSystem containing the chosen File.
 * @return int 0 if chosen File is locked by given UUID, -1 otherwise.
 */
int isLockedByFile(char *name, char *uuid, FileSystem *fs)
{
    File *file;

    PTHREAD_CHECK(READLOCK);
    if (hashTableGet(name, (void **)&file, *fs->filesTable) == -1)
    {
        UNLOCK;
        errno = EBADF;
        return -1;
    }
    PTHREAD_CHECK(UNLOCK);

    return fileIsLockedBy(file, uuid);
}

/**
 * @brief Removes the File described by fd from the FileSystem.
 *
 * @param fd a file descriptor gotten from opening the chosen File.
 * @param fs the FileSystem containing the chosen File.
 * @return int 0 on a success, -1 and sets errno on failure.
 */
int removeFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    void *saveptr = NULL;
    Metadata *meta;
    int i;
    HandlerRequest *request;

    // Check if the file is locked.
    if (isLockedByFile(fd->name, fd->uuid, fs) == -1)
    {
        errno = EINVAL;
        return -1;
    }

    PTHREAD_CHECK(LISTLOCK);
    CLEANUP_PTHREAD_CHECK(WRITELOCK, LISTUNLOCK);

    i = 0;

    // Gotta remove the file from the filesList.
    errno = 0;
    while (listScan((void **)&meta, &saveptr, fs->filesList) != -1)
    {
        if (!strcmp(meta->name, fd->name))
        {
            if (listRemove(i, (void **)&meta, fs->filesList) == -1)
            {
                perror("Error on listRemove");
            }
            break;
        }
        i++;
    }
    // listScan always misses the last element of the lsit.
    if (errno == EOF)
    {
        if (!strcmp(meta->name, fd->name))
        {

            if (listRemove(i, (void **)&meta, fs->filesList) == -1)
            {
                perror("Error on listRemove");
            }
        }
    }
    CLEANUP_PTHREAD_CHECK(LISTUNLOCK, UNLOCK);

    // Remove the actual file from the filesTable
    CLEANUP_ERROR_CHECK(hashTableRemove(fd->name, (void **)&file, *fs->filesTable), UNLOCK);

    // Update the FileSystem size
    atomicDec(getFileTrueSize(file), fs->curSize);
    atomicDec(1, fs->curN);

    // Free up the file's resources.
    fileDestroy(file);
    free(file);

    PTHREAD_CHECK(UNLOCK);

    atomicInc(1, fs->fsStats->remove);
    // We count a remove as an unlock to get an even number of locks/unlocks. Otherwise when one locks a file and then removes it the unlock is lost.
    atomicInc(1, fs->fsStats->unlock);

    UNSAFE_NULL_CHECK(request = malloc(sizeof(HandlerRequest)));
    handlerRequestInit(R_REMOVE, fd->name, fd->uuid, NULL, request);
    syncqueuePush((void *)request, fs->lockHandlerQueue);

    return 0;
}

/**
 * @brief Get the uncompressed size of the file of chosen name.
 *
 * @param pathname the name of the file.
 * @param fs the FileSystem containing the file
 * @return size_t the size of the file if it exists, 0 and sets errno otherwise.
 */
size_t getSize(const char *pathname, FileSystem *fs)
{
    File *file;
    size_t size;
    int tmp = 0;

    if (READLOCK != 0)
    {
        return 0;
    }

    if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
    {
        UNLOCK;
        errno = EINVAL;
        return 0;
    }

    errno = 0;
    size = getFileSize(file);
    tmp = errno;

    if (UNLOCK != 0)
    {
        return 0;
    }

    errno = tmp;
    return size;
}

/**
 * @brief Get the true size of the file of chosen name.
 *
 * @param pathname the name of the file.
 * @param fs the FileSystem containing the file
 * @return size_t the size of the file if it exists, 0 and sets errno otherwise.
 */

size_t getTrueSize(const char *pathname, FileSystem *fs)
{
    File *file;

    if (READLOCK != 0)
    {
        return 0;
    }

    if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
    {
        errno = EINVAL;
        return 0;
    }

    if (UNLOCK != 0)
    {
        return 0;
    }

    errno = 0;
    return getFileTrueSize(file);
}

/**
 * @brief Get the current size of the FileSystem.
 *
 * @param fs the FileSystem to query.
 * @return size_t the current number of files in the FileSystem.
 */
size_t getCurSize(FileSystem *fs)
{
    return atomicGet(fs->curSize);
}

/**
 * @brief Get the current number of files inside of the FileSystem.
 *
 * @param fs the FileSystem to query.
 * @return size_t the current number of files in the FileSystem.
 */
size_t getCurN(FileSystem *fs)
{
    return atomicGet(fs->curN);
}

/**
 * @brief Frees up space in the FileSystem, removing at least one file, and freeing enough space to have size free space. Returns the removed files in buf.
 *
 * @param size the amount of space to be freed.
 * @param buf where the removed file(s) will be stored. Should be freed.
 * @param fs the fileSystem to modify.
 * @return int the number of files removed, aka the size of buf on success. -1 and sets errno on failure.
 */
int freeSpace(size_t size, FileContainer **buf, FileSystem *fs)
{
    int64_t toFree;
    FileContainer *curFile;
    FileDescriptor *curFd;
    List tmpFiles;
    void *tmpBuf;
    size_t tmpSize, tmpTrueSize;
    const char *target;
    char log[500];
    int n = 0, i = 0;

    PTHREAD_CHECK(LISTLOCK);

    // Since we have no way of knowing how many Files will be removed beforehand, we put them in a dinamic list at first.
    listInit(&tmpFiles);
    toFree = size - (fs->maxSize - atomicGet(fs->curSize));
    if (size > fs->maxSize)
    {
        LISTUNLOCK;
        errno = EINVAL;
        return -1;
    }
    // If toFree < 0 we still always free at least one element.
    while (toFree > 0 || n <= 0)
    {
        // missPolicy gives us a chosen File, if one is available.
        if (missPolicy(&curFd, fs) == -1)
        {
            perror("MissPolicy error!");
            return -1;
        }

        target = curFd->name;

        tmpSize = getSize(target, fs);
        tmpTrueSize = getTrueSize(target, fs);

        // We put the file in a serializable FileContainer for the future.
        CLEANUP_CHECK(tmpBuf = malloc(tmpSize), NULL, { LISTUNLOCK; });
        CLEANUP_ERROR_CHECK(readFile(curFd, &tmpBuf, tmpSize, fs), { LISTUNLOCK; free(tmpBuf); });

        CLEANUP_CHECK(curFile = malloc(sizeof(FileContainer)), NULL, { LISTUNLOCK; free(tmpBuf); });
        CLEANUP_ERROR_CHECK(containerInit(tmpSize, tmpBuf, target, curFile), { LISTUNLOCK; free(tmpBuf); free(curFile); });
        free(tmpBuf);

        // And add it to the list.
        CLEANUP_ERROR_CHECK(listPush((void *)curFile, &tmpFiles), { LISTUNLOCK; });
        CLEANUP_ERROR_CHECK(removeFile(curFd, fs), { LISTUNLOCK; });

        fdDestroy(curFd);
        free(curFd);

        toFree -= tmpTrueSize;
        n++;
    }

    PTHREAD_CHECK(LISTUNLOCK);

    // Now that we have our list of FileContainers we put it in a buffer of the right size
    SAFE_NULL_CHECK(*buf = malloc(sizeof(FileContainer) * tmpFiles.size));

    for (i = 0; i < n; i++)
    {
        listPop((void **)(&curFile), &tmpFiles);
        (*buf)[i] = *curFile;
        free(curFile);
    }

    sprintf(log, ">To free:%ld >Removed:%d", size, n);
    logger(log, "CAPMISS");

    sprintf(log, ">curN:%ld >curSize:%ld", atomicGet(fs->curN), atomicGet(fs->curSize));
    logger(log, "SIZE");

    listDestroy(&tmpFiles);

    atomicInc(1, fs->fsStats->capMisses);

    // If no files were removed something went wrong.
    if (n == 0)
        n = -1;
    return n;
}

/**
 * @brief For internal use only, transforms a single Metadata in a human readable string which must be freed afterwards.
 *
 * @param el must point to a valid Metadata, the Metadata to print.
 * @return char* a string representing given Metadata, must be freed after use.
 */
char *_metadataPrinter(void *el)
{
    Metadata *data = (Metadata *)el;
    char *result;

    UNSAFE_NULL_CHECK(result = malloc(strlen(data->name) + 400));

    sprintf(result, "Name: %-30s \tSize: %-10ld \tCreation Time: %-10ld \tLast Access Time: %-10ld \tAccess Count: %ld", data->name, data->size, data->creationTime, data->lastAccess, data->numberAccesses);

    return result;
}

/**
 * @brief Prints a human readable account of the files in the given FileSystem.
 *
 * @param fs the FileSystem to print.
 */
void prettyPrintFiles(FileSystem *fs)
{
    LISTLOCK;
    puts("\n---------- FileSystem Contents ----------");
    printf("Number of files present:%-10ld\tSize of FileSystem:%-10ld\n", atomicGet(fs->curN), atomicGet(fs->curSize));
    puts("--------------- All Files ---------------");
    customPrintList(fs->filesList, &_metadataPrinter);
    puts("----------------- End ------------------\n");
    LISTUNLOCK;
}

/**
 * @brief Prints a human readable account of given FSStats.
 *
 * @param stats the FSStats to print.
 */
void prettyPrintStats(FSStats *stats)
{
    puts("\n---------- FileSystem Stats ----------");
    printf("Maximum number of files:%-10ld\tMaximum size of FileSystem:%-10ld\n", atomicGet(stats->maxN), atomicGet(stats->maxSize));
    printf("Number of files created:%-10ld\tNumber of capacity misses:%-10ld\n", atomicGet(stats->filesCreated), atomicGet(stats->capMisses));
    puts("---------- Operations Count ----------");
    printf("Open:%-10ld\tClose:%-10ld\tRead:%-10ld\tWrite:%-10ld\tAppend:%-10ld\n", atomicGet(stats->open), atomicGet(stats->close), atomicGet(stats->read), atomicGet(stats->write), atomicGet(stats->append));
    printf("ReadN:%-10ld\tLock:%-10ld\tUnlock:%-10ld\tRemove:%-10ld\n", atomicGet(stats->readN), atomicGet(stats->lock), atomicGet(stats->unlock), atomicGet(stats->remove));
    puts("---------------- End -----------------\n");
}

/**
 * @brief Initializes given FSStats.
 *
 * @param stats the FSStats to be initialized.
 * @return int 0 on success, -1 on failure.
 */
int statsInit(FSStats *stats)
{
    SAFE_NULL_CHECK(stats->maxSize = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->maxN = malloc(sizeof(AtomicInt)));

    SAFE_NULL_CHECK(stats->filesCreated = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->capMisses = malloc(sizeof(AtomicInt)));

    SAFE_NULL_CHECK(stats->open = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->close = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->read = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->write = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->append = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->readN = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->lock = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->unlock = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(stats->remove = malloc(sizeof(AtomicInt)));

    atomicInit(stats->maxSize);
    atomicInit(stats->maxN);

    atomicInit(stats->filesCreated);
    atomicInit(stats->capMisses);

    atomicInit(stats->open);
    atomicInit(stats->close);
    atomicInit(stats->read);
    atomicInit(stats->write);
    atomicInit(stats->append);
    atomicInit(stats->readN);
    atomicInit(stats->lock);
    atomicInit(stats->unlock);
    atomicInit(stats->remove);

    SAFE_NULL_CHECK(stats->sizeMtx = malloc(sizeof(pthread_mutex_t)));
    PTHREAD_CHECK(pthread_mutex_init(stats->sizeMtx, NULL));

    return 0;
}

/**
 * @brief Destroys given FSStats, freeing it's resources.
 *
 * @param stats the FSStats to be destroyed.
 */
void statsDestroy(FSStats *stats)
{
    atomicDestroy(stats->maxSize);
    atomicDestroy(stats->maxN);

    atomicDestroy(stats->filesCreated);
    atomicDestroy(stats->capMisses);

    atomicDestroy(stats->open);
    atomicDestroy(stats->close);
    atomicDestroy(stats->read);
    atomicDestroy(stats->write);
    atomicDestroy(stats->append);
    atomicDestroy(stats->readN);
    atomicDestroy(stats->lock);
    atomicDestroy(stats->unlock);
    atomicDestroy(stats->remove);

    free(stats->maxSize);
    free(stats->maxN);

    free(stats->filesCreated);
    free(stats->capMisses);

    free(stats->open);
    free(stats->close);
    free(stats->read);
    free(stats->write);
    free(stats->append);
    free(stats->readN);
    free(stats->lock);
    free(stats->unlock);
    free(stats->remove);

    pthread_mutex_destroy(stats->sizeMtx);
    free(stats->sizeMtx);
}

/**
 * @brief Updates FSStats maxSize and/or maxN, if given curSize or curN are bigger.
 *
 * @param stats the FSStats to update.
 * @param curSize the current size of the fileSystem.
 * @param curN the current number of files in the fileSystem.
 * @return int 0 on success, -1 on failure.
 */
int statsUpdateSize(FSStats *stats, size_t curSize, size_t curN)
{
    PTHREAD_CHECK(pthread_mutex_lock(stats->sizeMtx));
    if (curSize > atomicGet(stats->maxSize))
    {
        atomicPut(curSize, stats->maxSize);
    }
    if (curN > atomicGet(stats->maxN))
    {
        atomicPut(curN, stats->maxN);
    }
    PTHREAD_CHECK(pthread_mutex_unlock(stats->sizeMtx));

    return 0;
}