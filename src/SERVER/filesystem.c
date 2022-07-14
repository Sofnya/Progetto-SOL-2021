#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "SERVER/filesystem.h"
#include "COMMON/macros.h"
#include "SERVER/policy.h"
#include "SERVER/logging.h"

/**
 * @brief Initializes the given FileSystem with max number of files maxN and maximum memory occupation maxSize.
 *
 * @param maxN the maximum number of files in the filesystem, -1 means unlimited.
 * @param maxSize the maximum size the filesystem should take up, -1 means unlimited.
 * @param fs
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int fsInit(uint64_t maxN, uint64_t maxSize, int isCompressed, FileSystem *fs)
{
    pthread_mutexattr_t attr;
    fs->maxN = maxN;
    fs->maxSize = maxSize;

    SAFE_NULL_CHECK(fs->curN = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(fs->curSize = malloc(sizeof(AtomicInt)));

    atomicInit(fs->curN);
    atomicInit(fs->curSize);

    SAFE_NULL_CHECK(fs->filesList = malloc(sizeof(List)));
    SAFE_NULL_CHECK(fs->filesListMtx = malloc(sizeof(pthread_mutex_t)));

    SAFE_NULL_CHECK(fs->filesTable = malloc(sizeof(HashTable)));

    PTHREAD_CHECK(pthread_mutexattr_init(&attr));
    PTHREAD_CHECK(pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE));
    PTHREAD_CHECK(pthread_mutex_init(fs->filesListMtx, &attr));
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
            fileDestroy(curFile);
            free(curFile);
        }
        free(cur);
    }

    listDestroy(fs->filesList);
    hashTableDestroy(fs->filesTable);
    pthread_mutex_destroy(fs->filesListMtx);
    atomicDestroy(fs->curN);
    atomicDestroy(fs->curSize);

    free(fs->filesList);
    free(fs->filesTable);
    free(fs->filesListMtx);
    free(fs->curN);
    free(fs->curSize);
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
        PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));

        if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == 0)
        {
            PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));
            errno = EINVAL;
            return -1;
        }

        if (atomicGet(fs->curN) + 1 > fs->maxN)
        {
            pthread_mutex_unlock(fs->filesListMtx);
            errno = EOVERFLOW;
            return -1;
        }

        CLEANUP_CHECK(file = malloc(sizeof(File)), NULL, { pthread_mutex_unlock(fs->filesListMtx); });
        CLEANUP_CHECK(nameCopy = malloc(strlen(pathname) + 1), NULL, { pthread_mutex_unlock(fs->filesListMtx); });

        strcpy(nameCopy, pathname);

        CLEANUP_ERROR_CHECK(fileInit(pathname, fs->isCompressed, file), { pthread_mutex_unlock(fs->filesListMtx); });

        CLEANUP_ERROR_CHECK(hashTablePut(pathname, (void *)file, *fs->filesTable), { pthread_mutex_unlock(fs->filesListMtx); });

        CLEANUP_ERROR_CHECK(listAppend((void *)nameCopy, fs->filesList), { pthread_mutex_unlock(fs->filesListMtx); });

        atomicInc(1, fs->curN);
        atomicInc(getFileTrueSize(file), fs->curSize);

        sprintf(log, ">curN:%ld >curSize:%ld", atomicGet(fs->curN), atomicGet(fs->curSize));
        logger(log, "SIZE");

        PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        newFd->name = file->name;
        newFd->pid = getpid();
        newFd->flags = FI_READ | FI_WRITE | FI_APPEND;
    }
    else
    {

        if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
        {
            errno = EINVAL;
            return -1;
        }

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        newFd->name = file->name;
        newFd->pid = getpid();
        newFd->flags = FI_READ | FI_APPEND;
    }

    if (flags & O_LOCK)
    {
        CLEANUP_ERROR_CHECK(lockFile(newFd, fs), { free(newFd); });
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
int readFile(FileDescriptor *fd, void **buf, uint64_t size, FileSystem *fs)
{
    File *file;

    if (!(fd->flags & FI_READ))
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    if (getFileSize(file) > size)
    {
        errno = ENOBUFS;
        return -1;
    }

    return fileRead(*buf, size, file);
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
int writeFile(FileDescriptor *fd, void *buf, uint64_t size, FileSystem *fs)
{
    File *file;
    uint64_t oldSize, newSize;
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
    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));
    oldSize = getFileTrueSize(file);
    SAFE_ERROR_CHECK(fileWrite(buf, size, file));
    newSize = getFileTrueSize(file);

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
int appendToFile(FileDescriptor *fd, void *buf, uint64_t size, FileSystem *fs)
{
    File *file;
    uint64_t oldSize, newSize;
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

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    oldSize = getFileTrueSize(file);
    SAFE_ERROR_CHECK(fileAppend(buf, size, file));
    newSize = getFileTrueSize(file);
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
 * @param size Where bufs sizes will be stored, also an array.
 * @param fs The fileSystem to query.
 * @return int on success, the number of files actually read, on error -1 and sets errno.
 */
int readNFiles(int N, FileContainer **buf, FileSystem *fs)
{
    FileDescriptor *curFD;
    char *cur = NULL;
    int amount, i;
    uint64_t curSize;
    void *curBuffer;

    pthread_mutex_lock(fs->filesListMtx);

    amount = atomicGet(fs->curN);
    if (N > 0 && N < amount)
        amount = N;
    SAFE_NULL_CHECK(*buf = malloc(sizeof(FileContainer) * amount));

    for (i = 0; i < amount; i++)
    {
        CLEANUP_ERROR_CHECK(listGet(i, (void **)&cur, fs->filesList), pthread_mutex_unlock(fs->filesListMtx));

        curSize = getSize(cur, fs);
        CLEANUP_CHECK(curBuffer = malloc(curSize), NULL, pthread_mutex_unlock(fs->filesListMtx));

        CLEANUP_ERROR_CHECK(openFile(cur, 0, &curFD, fs), {pthread_mutex_unlock(fs->filesListMtx); free(curBuffer); });
        CLEANUP_ERROR_CHECK(readFile(curFD, &curBuffer, curSize, fs), {pthread_mutex_unlock(fs->filesListMtx); free(curBuffer); });
        CLEANUP_ERROR_CHECK(closeFile(curFD, fs), {pthread_mutex_unlock(fs->filesListMtx); free(curBuffer); });

        containerInit(curSize, curBuffer, cur, &((*buf)[i]));
        free(curBuffer);
    }
    pthread_mutex_unlock(fs->filesListMtx);

    return amount;
}

int lockFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    if (fd->flags & FI_LOCK)
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    SAFE_ERROR_CHECK(fileLock(file));
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

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    SAFE_ERROR_CHECK(fileUnlock(file));
    fd->flags &= ~FI_LOCK;

    return 0;
}

int tryLockFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    if (fd->flags & FI_LOCK)
    {
        return 0;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    if (fileTryLock(file) == -1)
    {
        return -1;
    }
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

    if (!((fd->flags & FI_LOCK)))
    {
        errno = EINVAL;
        return -1;
    }

    PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));
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
    PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));

    SAFE_ERROR_CHECK(hashTableRemove(fd->name, (void **)&file, *fs->filesTable));

    atomicDec(getFileTrueSize(file), fs->curSize);
    atomicDec(1, fs->curN);

    fileUnlock(file);
    fileDestroy(file);
    free(file);

    return 0;
}

/**
 * @brief Get the size of the file of chosen name.
 *
 * @param pathname the name of the file.
 * @param fs the fileSystem containing the file
 * @return uint64_t the size of the file if it exists, 0 and sets errno otherwise.
 */
uint64_t getSize(const char *pathname, FileSystem *fs)
{
    File *file;

    if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
    {
        errno = EINVAL;
        return 0;
    }

    errno = 0;
    return getFileSize(file);
}

/**
 * @brief Get the true size of the file of chosen name.
 *
 * @param pathname the name of the file.
 * @param fs the fileSystem containing the file
 * @return uint64_t the size of the file if it exists, 0 and sets errno otherwise.
 */

uint64_t getTrueSize(const char *pathname, FileSystem *fs)
{
    File *file;

    if (hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
    {
        errno = EINVAL;
        return 0;
    }

    errno = 0;
    return getFileTrueSize(file);
}

uint64_t getCurSize(FileSystem *fs)
{
    return atomicGet(fs->curSize);
}

uint64_t getCurN(FileSystem *fs)
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
int freeSpace(uint64_t size, FileContainer **buf, FileSystem *fs)
{
    int64_t toFree;
    FileContainer *curFile;
    FileDescriptor *curFd;
    List tmpFiles;
    void *tmpBuf;
    uint64_t tmpSize, tmpTrueSize;
    const char *target;
    char log[500];
    int n = 0, i = 0;

    PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));

    listInit(&tmpFiles);
    toFree = size - (fs->maxSize - atomicGet(fs->curSize));
    if (size > fs->maxSize)
    {
        pthread_mutex_unlock(fs->filesListMtx);
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

        CLEANUP_CHECK(tmpBuf = malloc(tmpSize), NULL, { pthread_mutex_unlock(fs->filesListMtx); });
        CLEANUP_ERROR_CHECK(readFile(curFd, &tmpBuf, tmpSize, fs), { pthread_mutex_unlock(fs->filesListMtx); free(tmpBuf); });

        CLEANUP_CHECK(curFile = malloc(sizeof(FileContainer)), NULL, { pthread_mutex_unlock(fs->filesListMtx); free(tmpBuf); });
        CLEANUP_ERROR_CHECK(containerInit(tmpSize, tmpBuf, target, curFile), { pthread_mutex_unlock(fs->filesListMtx); free(tmpBuf); free(curFile); });
        free(tmpBuf);

        CLEANUP_ERROR_CHECK(listPush((void *)curFile, &tmpFiles), { pthread_mutex_unlock(fs->filesListMtx); });
        CLEANUP_ERROR_CHECK(removeFile(curFd, fs), { pthread_mutex_unlock(fs->filesListMtx); });
        free(curFd);

        toFree -= tmpTrueSize;

        n++;
    }

    PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));

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
