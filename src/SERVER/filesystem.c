#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


#include "SERVER/filesystem.h"
#include "COMMON/macros.h"


/**
 * @brief Initializes the given FileSystem with max number of files maxN and maximum memory occupation maxSize.
 * 
 * @param maxN the maximum number of files in the filesystem, -1 means unlimited.
 * @param maxSize the maximum size the filesystem should take up, -1 means unlimited.
 * @param fs 
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int fsInit(uint64_t maxN, uint64_t maxSize, FileSystem *fs)
{
    fs->maxN = maxN;
    fs->maxSize = maxSize;

    SAFE_NULL_CHECK(fs->curN = malloc(sizeof(AtomicInt)));
    SAFE_NULL_CHECK(fs->curSize = malloc(sizeof(AtomicInt)));
    
    atomicInit(fs->curN);
    atomicInit(fs->curSize);

    
    SAFE_NULL_CHECK(fs->filesList = malloc(sizeof(List)));
    SAFE_NULL_CHECK(fs->filesListMtx = malloc(sizeof(pthread_mutex_t)));

    SAFE_NULL_CHECK(fs->filesTable = malloc(sizeof(HashTable)));

    
    PTHREAD_CHECK(pthread_mutex_init(fs->filesListMtx, NULL));
    SAFE_ERROR_CHECK(listInit(fs->filesList));
    
    if(maxN > 0){
        SAFE_ERROR_CHECK(hashTableInit(maxN * 2, fs->filesTable));
    } 
    else
    {
        SAFE_ERROR_CHECK(hashTableInit(4096, fs->filesTable));
    }

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

    puts("Destroying FileSystem!");
    while(listSize(*fs->filesList) > 0)
    {
        UNSAFE_NULL_CHECK(curFile = malloc(sizeof(File)));
        listPop((void **)&cur, fs->filesList);
        puts(cur);
        puts("Just a print...");
        if(hashTableRemove(cur, (void **)&curFile, *fs->filesTable) == 0)
        {
            fileDestroy(curFile);
            free(curFile);
        }
    }

    puts("Some more prints");
    listDestroy(fs->filesList);
    hashTableDestroy(fs->filesTable);
    pthread_mutex_destroy(fs->filesListMtx);
    atomicDestroy(fs->curN);
    atomicDestroy(fs->curSize);

    puts("Now freeing!");

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
int openFile(const char* pathname, int flags, FileDescriptor **fd, FileSystem *fs)
{
    File *file;
    FileDescriptor *newFd;
    int res;


    if(pathname == NULL) puts("\n\n\n\n\n\n\n\n\nSomething very wrong!");
    res = hashTableGet(pathname, (void **)&file, *fs->filesTable);

    if(flags & O_CREATE)
    {
        if(res == 0) 
        {
            errno = EINVAL;
            return -1;
        }
        
        SAFE_NULL_CHECK(file = malloc(sizeof(File)));
        ERROR_CHECK(fileInit(pathname, file));

        ERROR_CHECK(hashTablePut(pathname, (void *)file, *fs->filesTable));
        
        
        PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));

        ERROR_CHECK(listAppend((void *)file->name, fs->filesList));
        puts("Creating file:");
        puts(file->name);

        PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));

        atomicInc(1, fs->curN);

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        newFd->name = file->name;
        newFd->pid = getpid();
        newFd->flags = FI_READ | FI_WRITE;

        *fd = newFd;


        return 0;

    }
    else
    {
        if(res == -1)
        {
            errno = EINVAL;
            return -1;
        }

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        newFd->name = file->name;
        newFd->pid = getpid();
        newFd->flags = FI_READ | FI_WRITE;

        *fd = newFd;

        return 0;
    }
}


int closeFile(FileDescriptor *fd, FileSystem *fs)
{
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
int readFile(FileDescriptor *fd, void** buf, size_t size, FileSystem *fs)
{
    File *file;

    if(!(fd->flags & FI_READ))
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    if(getFileSize(file) > size)
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
int writeFile(FileDescriptor *fd, void *buf, size_t size, FileSystem *fs)
{
    File *file;
    uint64_t oldSize;

    if(!(fd->flags & FI_WRITE))
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    SAFE_ERROR_CHECK(fileLock(file));
    oldSize = getFileSize(file);
    SAFE_ERROR_CHECK(fileWrite(buf, size, file));
    SAFE_ERROR_CHECK(fileUnlock(file));
    
    atomicInc(size - oldSize, fs->curSize);

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
int appendToFile(FileDescriptor *fd, void* buf, size_t size, FileSystem *fs)
{
    File *file;


    if(!(fd->flags & FI_WRITE))
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    SAFE_ERROR_CHECK(fileLock(file));
    SAFE_ERROR_CHECK(fileAppend(buf, size, file));
    SAFE_ERROR_CHECK(fileUnlock(file));


    atomicInc(size, fs->curSize);

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

    if(!(fd->flags & FI_WRITE))
    {
        errno = EINVAL;
        return -1;
    }


    
    PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));
    i = 0;
    puts("Current list state:");
    while(listScan((void **)&name, &saveptr, fs->filesList) != -1)
    {
        puts(name);
        if(!strcmp(name,fd->name))
        {
            puts("Removing file:");
            puts(name);
            listRemove(i, NULL, fs->filesList);
            break;
        }
        i++;
    }
    PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));


    SAFE_ERROR_CHECK(hashTableRemove(fd->name, (void **)&file, *fs->filesTable));
    

    //SAFE_ERROR_CHECK(fileLock(file));
    atomicDec(getFileSize(file), fs->curSize);
    atomicDec(1, fs->curN);

    fileDestroy(file);

    return 0;
}

/**
 * @brief Get the size of the file of chosen name.
 * 
 * @param pathname the name of the file.
 * @param fs the fileSystem containing the file
 * @return uint64_t the size of the file if it exists, 0 and sets errno otherwise.
 */
uint64_t getSize(const char* pathname, FileSystem *fs)
{
    File *file;

    if(hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
    { 
        errno = EINVAL;
        return 0;
    }

    errno = 0;
    return getFileSize(file);
}
