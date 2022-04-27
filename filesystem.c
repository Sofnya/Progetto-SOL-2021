#include <errno.h>
#include <stdlib.h>


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

    fs->curN = 0;
    fs->curSize = 0;

    
    SAFE_NULL_CHECK(fs->filesList = malloc(sizeof(List)));
    SAFE_NULL_CHECK(fs->filesTable = malloc(sizeof(HashTable)));

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
    int i, len;
    char *cur;
    File *curFile;

    while(listSize(*fs->filesList) > 0)
    {
        listPop(&cur, fs->filesList);
        hashTableRemove(cur, curFile, *fs->filesTable);
        
        fileDestroy(curFile);
        free(curFile);
        free(cur);
    }
    listDestroy(fs->filesList);
    hashTableDestroy(fs->filesTable);

    free(fs->filesList);
    free(fs->filesTable);

}


int openFile(const char* pathname, int flags, FileSystem *fs)
{
    File *file;
    int res;


    res = hashTableGet(pathname, file, *fs->filesTable);

    if(flags & O_CREATE)
    {
        if(res == 0) 
        {
            errno = EINVAL;
            return -1;
        }
        
        SAFE_NULL_CHECK(file = malloc(sizeof(File)));
        fileInit(pathname, file);
        fileOpen(flags, file);
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
