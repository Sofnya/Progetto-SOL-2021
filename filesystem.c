#include <errno.h>
#include <stdlib.h>
#include <unistd.h>


#include "filesystem.h"
#include "macros.h"


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

    atomicInit(&fs->curN);
    atomicInit(&fs->curSize);

    
    SAFE_NULL_CHECK(fs->filesList = malloc(sizeof(List)));
    SAFE_NULL_CHECK(fs->openFiles = malloc(sizeof(List)));
    SAFE_NULL_CHECK(fs->filesTable = malloc(sizeof(HashTable)));

    SAFE_ERROR_CHECK(listInit(fs->filesList));
    SAFE_ERROR_CHECK(listInit(fs->openFiles));
    
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
    int i, len;
    char *cur;
    File *curFile;

    while(listSize(*fs->openFiles) > 0)
    {
        listPop(&cur, fs->openFiles);
        free(cur);
    }
    listDestroy(fs->openFiles);

    while(listSize(*fs->filesList) > 0)
    {
        listPop(&cur, fs->filesList);
        hashTableRemove(cur, curFile, *fs->filesTable);
        
        fileDestroy(curFile);
        free(curFile);
    }
    listDestroy(fs->filesList);
    hashTableDestroy(fs->filesTable);

    free(fs->filesList);
    free(fs->filesTable);

}


int openFile(const char* pathname, int flags, FileSystem *fs)
{
    File *file;
    FileDescriptor *fd;
    int res;


    res = hashTableGet(pathname, file, *fs->filesTable);

    if(flags & O_CREATE)
    {
        if(res == 0) 
        {
            errno = EINVAL;
            return -1;
        }
        
        PTHREAD_CHECK(pthread_mutex_lock(fs->filesListMtx));
        PTHREAD_CHECK(pthread_mutex_lock(fs->openFilesMtx));
        
        SAFE_NULL_CHECK(file = malloc(sizeof(File)));
        ERROR_CHECK(fileInit(pathname, file));

        ERROR_CHECK(hashTablePut(pathname, (void *)file, *fs->filesTable));
        ERROR_CHECK(listAppend(file->name, fs->filesList));

        atomicInc(1, &fs->curN);

        SAFE_NULL_CHECK(fd = malloc(sizeof(FileDescriptor)));
        fd->name = file->name;
        fd->pid = getpid();
        fd->flags = flags;

        listAppend((void *)fd, fs->openFiles);

        return 0;
        PTHREAD_CHECK(pthread_mutex_unlock(fs->openFilesMtx));
        PTHREAD_CHECK(pthread_mutex_unlock(fs->filesListMtx));

    }
    else
    {
        if(res == -1)
        {
            errno = EINVAL;
            return -1;
        }
    }
}


int closeFile(const char* pathname, FileSystem *fs);
