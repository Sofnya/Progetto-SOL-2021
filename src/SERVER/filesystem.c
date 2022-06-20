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
        }
        free(curFile);
        free(cur);
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
    char *nameCopy;
    FileDescriptor *newFd;


    if(pathname == NULL) puts("\n\n\n\n\n\n\n\n\nSomething very wrong!");
    

    if(flags & O_CREATE)
    {
        PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));
        if(hashTableGet(pathname, (void **)&file, *fs->filesTable) == 0) 
        {

            PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));
            errno = EINVAL;
            return -1;
        }
        
        SAFE_NULL_CHECK(file = malloc(sizeof(File)));
        SAFE_NULL_CHECK(nameCopy = malloc(strlen(pathname) + 1));

        strcpy(nameCopy, pathname);
        
        ERROR_CHECK(fileInit(pathname, file));

        ERROR_CHECK(hashTablePut(pathname, (void *)file, *fs->filesTable));


        ERROR_CHECK(listAppend((void *)nameCopy, fs->filesList));
        puts("Creating file:");
        puts(nameCopy);

        PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));

        atomicInc(1, fs->curN);

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        newFd->name = file->name;
        newFd->pid = getpid();
        newFd->flags = FI_READ | FI_WRITE | FI_APPEND;
    }
    else
    {
        
        if(hashTableGet(pathname, (void **)&file, *fs->filesTable) == -1)
        {
            errno = EINVAL;
            return -1;
        }

        SAFE_NULL_CHECK(newFd = malloc(sizeof(FileDescriptor)));
        newFd->name = file->name;
        newFd->pid = getpid();
        newFd->flags = FI_READ | FI_APPEND;
    }

    if(flags & O_LOCK) 
    {
        lockFile(newFd, fs);
    }
    *fd = newFd;

    return 0;
}


int closeFile(FileDescriptor *fd, FileSystem *fs)
{
    if(fd->flags & FI_LOCK)
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

    if(!((fd->flags & FI_WRITE) && (fd->flags & FI_LOCK)))
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    oldSize = getFileSize(file);
    SAFE_ERROR_CHECK(fileWrite(buf, size, file));
    
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


    if(!((fd->flags & FI_APPEND) && (fd->flags & FI_LOCK)))
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    SAFE_ERROR_CHECK(fileAppend(buf, size, file));


    atomicInc(size, fs->curSize);

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
    
    puts("Lock acquired");

    amount = atomicGet(fs->curN);
    if(N > 0 && N < amount) amount = N;
    SAFE_NULL_CHECK(*buf = malloc(sizeof(FileContainer) * amount));

    puts("Initial malloc done");
    for(i = 0; i < amount; i++)
    {
        printf("Reading file:%d\n", i);
        SAFE_ERROR_CHECK(listGet(i, (void **)&cur, fs->filesList));

        puts("ListGet done!!");
        curSize = getSize(cur, fs);
        SAFE_NULL_CHECK(curBuffer = malloc(curSize));

        SAFE_ERROR_CHECK(openFile(cur, 0, &curFD, fs));
        puts("fileOpened!");        
        SAFE_ERROR_CHECK(readFile(curFD, curBuffer, curSize, fs));
        SAFE_ERROR_CHECK(closeFile(curFD, fs));

        puts("Initializing container..");
        containerInit(curSize, curBuffer, cur, buf[i]);
        
        free(curBuffer);
    }
    pthread_mutex_unlock(fs->filesListMtx);

    puts("Ok done with readN!");
    return amount;
}




int lockFile(FileDescriptor *fd, FileSystem *fs)
{
    File *file;
    if(fd->flags & FI_LOCK)
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
    if(!(fd->flags & FI_LOCK))
    {
        errno = EINVAL;
        return -1;
    }

    SAFE_ERROR_CHECK(hashTableGet(fd->name, (void **)&file, *fs->filesTable));

    SAFE_ERROR_CHECK(fileUnlock(file));
    fd->flags &= ~FI_LOCK;

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

    puts("Calling removefile.");
    printf("Name: |%s|\n", fd->name);

    if(!((fd->flags & FI_WRITE) && (fd->flags & FI_LOCK)))
    {
        errno = EINVAL;
        return -1;
    }


    
    PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));
    i = 0;
    puts("Current list state:");
    printf("Len:%lld\n", fs->filesList->size);
    while(listScan((void **)&name, &saveptr, fs->filesList) != -1)
    {
        printf("%d:%p->|%s|\n", i, name, name);
        if(!strcmp(name,fd->name))
        {
            puts("Removing file:");
            puts(name);
            listRemove(i, (void **)&name, fs->filesList);
            printf("Actually removed: |%s|\n", name);
            free(name);
            break;
        }
        i++;
    }
    PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));


    SAFE_ERROR_CHECK(hashTableRemove(fd->name, (void **)&file, *fs->filesTable));
    

    atomicDec(getFileSize(file), fs->curSize);
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
