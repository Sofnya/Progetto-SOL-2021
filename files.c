#include "files.h"
#include <string.h>
#include <stdlib.h>

#include "COMMON/macros.h"


/**
 * @brief Initializes given file with the correct name.
 * 
 * @param name the pathname of the file
 * @param file the file to initialize
 * @return int 0 on success, -1 and sets errno otherwise
 */
int fileInit(const char *name, File *file)
{
    size_t nameSize;

    nameSize = strlen(name) + 1;
    SAFE_NULL_CHECK(file->name = malloc(sizeof(char) * nameSize));
    strcpy(file->name, name);

    file->size = 0;
    file->content = NULL;
    SAFE_NULL_CHECK(file->mtx = malloc(sizeof(pthread_mutex_t)));
    PTHREAD_CHECK(pthread_mutex_init(file->mtx, NULL));
    return 0;
}

/**
 * @brief Destroys given file, freeing it's memory.
 * 
 * @param file the file to be destroyed
 */
void fileDestroy(File *file)
{
    free(file->name);
    free(file->content);

    PTHREAD_CHECK(pthread_mutex_destroy(file->mtx));
    free(file->mtx);
}

/**
 * @brief Writes the given content to the file.
 * 
 * @param content the content to be written.
 * @param size the size of content.
 * @param file the file to be modified.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileWrite(const void *content, uint64_t size, File *file)
{
    UNSAFE_NULL_CHECK(file->content = realloc(file->content, size));
    file->size = size;

    memcpy(file->content, content, size);

    return 0;
}

/**
 * @brief Appends the given content at the end of the file.
 * 
 * @param content the content to be written.
 * @param size the size of the new content.
 * @param file the file to be modified.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileAppend(const void *content, uint64_t size, File *file)
{
    UNSAFE_NULL_CHECK(file->content = realloc(file->content, size + file->size));
    memcpy(file->content + file->size, content, size);

    file->size += size;

    return 0;
}

/**
 * @brief will read up to bufsize bytes from the files contents inside of the given buf.
 * 
 * @param buf where the contents will be returned.
 * @param bufsize the size of buf.
 * @param file the file to be read.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int fileRead(void *buf, uint64_t bufsize, File *file)
{
    uint64_t n;

    if(bufsize < file->size) n = bufsize;
    else n = file->size;

    memcpy(buf, file->content, n);

    return 0;
}


/**
 * @brief Trys to lock the file, without blocking.
 * 
 * @param file 
 * @return int 0 if succesfull, -1 and sets errno on if file is already locked.
 */
int fileTryLock(File *file)
{
    int res;

    res = pthread_mutex_trylock(file->mtx);
    if(res != 0) 
    {
        errno = res;
        return -1;    
    }
    return 0;
}


/**
 * @brief Locks the file.
 * 
 * @param file 
 * @return int 
 */
int fileLock(File *file)
{
    PTHREAD_CHECK(pthread_mutex_lock(file->mtx));
    return 0;
}


/**
 * @brief Unlocks the file.
 * 
 * @param file 
 * @return int 
 */
int fileUnlock(File *file)
{
    PTHREAD_CHECK(pthread_mutex_unlock(file->mtx));
    return 0;
}


/**
 * @brief Get the File's size.
 * 
 * @param file 
 * @return uint64_t 
 */
uint64_t getFileSize(File *file)
{
    return file->size;
}

/**
 * @brief Get the File's name.
 * 
 * @param file 
 * @return char* 
 */
char *getFileName(File *file)
{
    return file->name;
}

