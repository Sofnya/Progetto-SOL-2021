#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "SERVER/filesystem.h"
#include "COMMON/macros.h"
#include "SERVER/policy.h"
#include "SERVER/logging.h"

#define READLOCK readLock(fs->rwLock, __LINE__, __func__)
#define WRITELOCK writeLock(fs->rwLock, __LINE__, __func__)
#define UNLOCK unlock(fs->rwLock, __LINE__, __func__)
#define LISTLOCK listLock(fs->filesListMtx, __LINE__, __func__)
#define LISTUNLOCK listUnlock(fs->filesListMtx, __LINE__, __func__)

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
 * @brief Initializes the given FileSystem with max number of files maxN and maximum memory occupation maxSize.
 *
 * @param maxN the maximum number of files in the filesystem, -1 means unlimited.
 * @param maxSize the maximum size the filesystem should take up, -1 means unlimited.
 * @param fs
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

    PTHREAD_CHECK(pthread_mutexattr_init(&attr));
    PTHREAD_CHECK(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE));
    PTHREAD_CHECK(pthread_mutex_init(fs->filesListMtx, &attr));
    PTHREAD_CHECK(pthread_rwlockattr_init(&rwAttr));
    PTHREAD_CHECK(pthread_rwlock_init(fs->rwLock, &rwAttr));
    PTHREAD_CHECK(pthread_mutexattr_destroy(&attr));

    SAFE_ERROR_CHECK(listInit(fs->filesList));

    if (maxN > 0)
    {
        SAFE_ERROR_CHECK(hashTableInit(maxN * 2, fs->filesTable));
    }
    else
    {
        SAFE_ERROR_CHECK(hashTableInit(4096, fs->filesTable));
    }

    fs->isCompressed = isCompressed;

    return 0;
}

/**
 * @brief Destroys the given FileSystem, cleaning up it's memory.
 *
 * @param fs the FileSystem to be destroyed.
 */
void fsDestroy(FileSystem *fs)
{
    char *cur;
    File *curFile;

    logger("Destroying fileSystem", "STATUS");
    while (listSize(*fs->filesList) > 0)
    {
        listPop((void **)&cur, fs->filesList);
        if (hashTableRemove(cur, (void **)&curFile, *fs->filesTable) == 0)
        {
            fileLock(curFile);
            fileDestroy(curFile);
            free(curFile);
        }
        free(cur);
    }

    listDestroy(fs->filesList);
    hashTableDestroy(fs->filesTable);
    pthread_mutex_destroy(fs->filesListMtx);
    pthread_rwlock_destroy(fs->rwLock);
    atomicDestroy(fs->curN);
    atomicDestroy(fs->curSize);

    free(fs->filesList);
    free(fs->filesTable);
    free(fs->filesListMtx);
    free(fs->rwLock);
    free(fs->curN);
    free(fs->curSize);
}

int fdInit(const char *name, pid_t pid, int flags, FileDescriptor *fd)
{
    size_t nameLen = strlen(name) + 1;

    fd->pid = pid;
    fd->flags = flags;
    SAFE_NULL_CHECK(fd->name = malloc(nameLen * sizeof(char)));

    strcpy(fd->name, name);

    return 0;
}

void fdDestroy(FileDescriptor *fd)
{
    free(fd->name);
    fd->name = NULL;
}

/**
 * @brief Opens a file, returning a FileDescriptor with given flags.
 *
 * @param pathname
 * @param flags
 * @param fd
 * @param fs
 * @return int
 */
int openFile(char *pathname, int flags, FileDescriptor **fd, FileSystem *fs)
{
    File *file;
    char *nameCopy;
    FileDescriptor *newFd;
    char log[500];

    if (pathname == NULL)
        puts("\n\n\n\n\n\n\n\n\nSomething very wrong!");

    if (flags & O_CREATE)
    {
        PTHREAD_CHECK(LISTLOCK);
        PTHREAD_CHECK(WRITELOCK);

        if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == 0)
        {
            UNLOCK;
            PTHREAD_CHECK(LISTUNLOCK);
            errno = EINVAL;
            return -1;
        }

        if (atomicGet(fs->curN) + 1 > fs->maxN)
        {
            UNLOCK;
            LISTUNLOCK;

            errno = EOVERFLOW;
            return -1;
        }

        CLEANUP_CHECK(file = malloc(sizeof(File)), NULL, {
            UNLOCK;
            LISTUNLOCK;
        });
        CLEANUP_CHECK(nameCopy = malloc(strlen(pathname) + 1), NULL, {
            UNLOCK;
            LISTUNLOCK;
        });

        strcpy(nameCopy, pathname);

        CLEANUP_ERROR_CHECK(fileInit(pathname, fs->isCompressed, file), {
            UNLOCK;
            LISTUNLOCK;
        });

        CLEANUP_ERROR_CHECK(hashTablePut(pathname, (void *)file, *fs->filesTable), {
            UNLOCK;
            LISTUNLOCK;
        });

        CLEANUP_ERROR_CHECK(listAppend((void *)nameCopy, fs->filesList), {
            UNLOCK;
            LISTUNLOCK;
        });

        atomicInc(1, fs->curN);
        atomicInc(getFileTrueSize(file), fs->curSize);

        sprintf(log, ">curN:%ld >curSize:%ld", atomicGet(fs->curN), atomicGet(fs->curSize));
        logger(log, "SIZE");

        PTHREAD_CHECK(UNLOCK);
        PTHREAD_CHECK(LISTUNLOCK);

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        fdInit(file->name, getpid(), FI_READ | FI_WRITE | FI_APPEND, newFd);
    }
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
        fdInit(file->name, getpid(), FI_READ | FI_APPEND, newFd);
        PTHREAD_CHECK(UNLOCK);
    }

    if (flags & O_LOCK)
    {
        CLEANUP_ERROR_CHECK(lockFile(newFd, fs), { fdDestroy(newFd); free(newFd); });
    }
    *fd = newFd;

    return 0;
}

int closeFile(FileDescriptor *fd, FileSystem *fs)
{
    if (fd->flags & FI_LOCK)
    {
        unlockFile(fd, fs);
    }

    fdDestroy(fd);
    free(fd);
    return 0;
}

/**
 * @brief Reads the contents of the file in fd, and puts them in buf.
 *
 * @param fd a file descriptor gotten from opening the chosen file.
 * @param buf the buffer in which the file's contents will be stored.
 * @param size the size of buf, if not large enough to hold all of the file's contents an error will be returned.
 * @param fs the FileSystem containing the chosen file.
 * @return int 0 on a success, -1 and sets errno on failure.
 */
int readFile(FileDescriptor *fd, void **buf, size_t size, FileSystem *fs)
{
    File *file;
    int success, tmpErr;

    if (!(fd->flags & FI_READ))
    {
        errno = EINVAL;
        return -1;
    }

    PTHREAD_CHECK(READLOCK);

    CLEANUP_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable), UNLOCK);

    if (getFileSize(file) > size)
    {
        PTHREAD_CHECK(UNLOCK);
        errno = ENOBUFS;
        return -1;
    }

    success = fileRead(*buf, size, file);
    tmpErr = errno;

    PTHREAD_CHECK(UNLOCK);
    errno = tmpErr;
    return success;
}

/**
 * @brief Writes the contents of buffer buf to the file described by fd.
 *
 * @param fd a file descriptor gotten from opening the chosen file.
 * @param buf the buffer from which the contents will be copyed.
 * @param size the size of buf, the file will be resized accordingly.
 * @param fs the FileSystem containing the chosen file.
 * @return int 0 on a success, -1 and sets errno on failure.
 */
int writeFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs)
{
    File *file;
    size_t oldSize, newSize;
    char log[500];
    if (!((fd->flags & FI_WRITE) && (fd->flags & FI_LOCK)))
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

    return 0;
}

/**
 * @brief Appends the contents of buf to the file described by fd.
 *
 * @param fd a file descriptor gotten from opening the chosen file.
 * @param buf the buffer from which the contents will be copyed.
 * @param size the size of buf.
 * @param fs the FileSystem containing the chosen file.
 * @return int 0 on a success, -1 and sets errno on failure.
 */
int appendToFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs)
{
    File *file;
    size_t oldSize, newSize;
    char log[500];

    if (!((fd->flags & FI_APPEND) && (fd->flags & FI_LOCK)))
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
    CLEANUP_ERROR_CHECK(fileAppend(buf, size, file), { UNLOCK; });
    newSize = getFileTrueSize(file);
    PTHREAD_CHECK(UNLOCK);

    atomicInc(newSize - oldSize, fs->curSize);

    sprintf(log, ">curN:%ld >curSize:%ld", atomicGet(fs->curN), atomicGet(fs->curSize));
    logger(log, "SIZE");

    return 0;
}

/**
 * @brief Reads N files in the given buffer. Buf and size will be malloced, so remember to free them after use.
 *
 * @param N The number of files to read, if N < 0 reads all of the servers files.
 * @param buf Where the files will be stored, will become an array of buffers.
 * @param fs The fileSystem to query.
 * @return int on success, the number of files actually read, on error -1 and sets errno.
 */
int readNFiles(int N, FileContainer **buf, FileSystem *fs)
{
    FileDescriptor *curFD;
    char *cur = NULL;
    int amount, i = 0;
    size_t curSize;
    void *curBuffer;

    PTHREAD_CHECK(LISTLOCK);

    amount = atomicGet(fs->curN);
    if (N > 0 && N < amount)
        amount = N;
    SAFE_NULL_CHECK(*buf = malloc(sizeof(FileContainer) * amount));

    while (i < amount)
    {
        CLEANUP_ERROR_CHECK(listGet(i, (void **)&cur, fs->filesList), { LISTUNLOCK; free(buf); });

        curSize = getSize(cur, fs);
        CLEANUP_CHECK(curBuffer = malloc(curSize), NULL, { LISTUNLOCK; });

        if (openFile(cur, 0, &curFD, fs) == -1)
        {
            free(curBuffer);
            continue;
        }
        if (readFile(curFD, &curBuffer, curSize, fs) == -1)
        {
            free(curBuffer);
            free(curFD);
            continue;
        }
        if (closeFile(curFD, fs) == -1)
        {
            free(curBuffer);
            free(curFD);
            continue;
        }

        CLEANUP_ERROR_CHECK(containerInit(curSize, curBuffer, cur, &((*buf)[i])), {LISTUNLOCK;  free(buf); });
        free(curBuffer);
        i++;
    }
    PTHREAD_CHECK(LISTUNLOCK);

    return amount;
}

int lockFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    int tmp;
    if (fd->flags & FI_LOCK)
    {
        errno = EINVAL;
        return -1;
    }

    PTHREAD_CHECK(READLOCK);

    CLEANUP_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable), UNLOCK);

    PTHREAD_CHECK(UNLOCK);
    tmp = fileLock(file);
    if (tmp != 0)
    {
        return tmp;
    }

    fd->flags |= FI_LOCK;
    return 0;
}

int unlockFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    if (!(fd->flags & FI_LOCK))
    {
        errno = EINVAL;
        return -1;
    }

    PTHREAD_CHECK(READLOCK);

    CLEANUP_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable), UNLOCK);

    CLEANUP_ERROR_CHECK(fileUnlock(file), UNLOCK);
    PTHREAD_CHECK(UNLOCK);
    fd->flags &= ~FI_LOCK;

    return 0;
}

int tryLockFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    int tmpErrno;
    if (fd->flags & FI_LOCK)
    {
        return 0;
    }

    PTHREAD_CHECK(READLOCK);

    CLEANUP_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable), UNLOCK);

    if (fileTryLock(file) == -1)
    {
        tmpErrno = errno;
        PTHREAD_CHECK(UNLOCK);
        errno = tmpErrno;
        return -1;
    }

    PTHREAD_CHECK(UNLOCK);
    fd->flags |= FI_LOCK;

    return 0;
}

/**
 * @brief Removes the file described by fd from the FileSystem.
 *
 * @param fd a file descriptor gotten from opening the chosen file.
 * @param fs the FileSystem containing the chosen file.
 * @return int 0 on a success, -1 and sets errno on failure.
 */
int removeFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    void *saveptr = NULL;
    char *name;
    int i;

    if (!(fd->flags & FI_LOCK))
    {
        errno = EINVAL;
        return -1;
    }

    PTHREAD_CHECK(LISTLOCK);
    CLEANUP_PTHREAD_CHECK(WRITELOCK, LISTUNLOCK);

    i = 0;

    while (listScan((void **)&name, &saveptr, fs->filesList) != -1)
    {
        if (!strcmp(name, fd->name))
        {
            if (listRemove(i, (void **)&name, fs->filesList) == -1)
            {
                perror("Error on listRemove");
            }

            free(name);
            break;
        }
        i++;
    }
    if (errno == EOF)
    {
        if (!strcmp(name, fd->name))
        {

            if (listRemove(i, (void **)&name, fs->filesList) == -1)
            {
                perror("Error on listRemove");
            }

            free(name);
        }
    }
    CLEANUP_PTHREAD_CHECK(LISTUNLOCK, UNLOCK);

    CLEANUP_ERROR_CHECK(hashTableRemove(fd->name, (void **)&file, *fs->filesTable), UNLOCK);

    atomicDec(getFileTrueSize(file), fs->curSize);
    atomicDec(1, fs->curN);

    fileDestroy(file);
    free(file);

    PTHREAD_CHECK(UNLOCK);

    return 0;
}

/**
 * @brief Get the size of the file of chosen name.
 *
 * @param pathname the name of the file.
 * @param fs the fileSystem containing the file
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
 * @param fs the fileSystem containing the file
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

size_t getCurSize(FileSystem *fs)
{
    return atomicGet(fs->curSize);
}

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
        if (missPolicy(&curFd, fs) == -1)
        {
            perror("MissPolicy error!");
            break;
        }

        target = curFd->name;

        tmpSize = getSize(target, fs);
        tmpTrueSize = getTrueSize(target, fs);

        CLEANUP_CHECK(tmpBuf = malloc(tmpSize), NULL, { LISTUNLOCK; });
        CLEANUP_ERROR_CHECK(readFile(curFd, &tmpBuf, tmpSize, fs), { LISTUNLOCK; free(tmpBuf); });

        CLEANUP_CHECK(curFile = malloc(sizeof(FileContainer)), NULL, { LISTUNLOCK; free(tmpBuf); });
        CLEANUP_ERROR_CHECK(containerInit(tmpSize, tmpBuf, target, curFile), { LISTUNLOCK; free(tmpBuf); free(curFile); });
        free(tmpBuf);

        CLEANUP_ERROR_CHECK(listPush((void *)curFile, &tmpFiles), { LISTUNLOCK; });
        CLEANUP_ERROR_CHECK(removeFile(curFd, fs), { LISTUNLOCK; });

        fdDestroy(curFd);
        free(curFd);

        toFree -= tmpTrueSize;

        n++;
    }

    PTHREAD_CHECK(LISTUNLOCK);
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

    if (n == 0)
        n = -1;
    return n;
}
