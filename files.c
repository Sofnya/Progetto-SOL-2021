#include "files.h"
#include <string.h>
#include <stdlib.h>

#include "macros.h"


/**
 * @brief Initializes given file with the correct name.
 * 
 * @param name the pathname of the file
 * @param file the file to initialize
 * @return int 0 on success, an errorcode otherwise
 */
int fileInit(char *name, File *file)
{
    size_t nameSize;

    nameSize = strlen(name) + 1;
    file->name = malloc(sizeof(char)* nameSize);
    strcpy(file->name, name);

    file->size = 0;
    file->content = NULL;

    file->flags = 0;

    PTHREAD_CHECK(pthread_mutex_init(file->mtx, NULL));
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
}

/**
 * @brief Writes the given content of size size to the file, appends to the contents if O_APPEND is set in the files flags.
 * 
 * @param content the content to be written.
 * @param size how much content should be written.
 * @param file the file to be modified.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileWrite(void *content, uint64_t size, File *file)
{
    if(!((file->flags & O_LOCK) && (file->flags & O_WRITE) && (file->flags & O_OPEN))) 
    {
        errno = EACCES;
        return -1;
    }
    if(file->flags == O_APPEND)
    {
        UNSAFE_NULL_CHECK(file->content = realloc(file->content, size + file->size));
        memcpy(content, size, file->content + file->size);

        file->size += size;

        return 0;
    }

    UNSAFE_NULL_CHECK(file->content = realloc(file->content, size));
    file->size = size;


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

    if(!((file->flags & O_LOCK) && (file->flags & O_READ) && (file->flags & O_OPEN))) 
    {
        errno = EACCES;
        return -1;
    }

    if(bufsize < file->size) n = bufsize;
    else n = file->size;

    memcpy(buf, file->content, n);

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
    file->flags |= O_LOCK;
}


/**
 * @brief Unlocks the file.
 * 
 * @param file 
 * @return int 
 */
int fileUnlock(File *file)
{
    file->flags &= ~O_LOCK;
    PTHREAD_CHECK(pthread_mutex_unlock(file->mtx));
}

/**
 * @brief Open the file with given flags.
 * 
 * @param flags 
 * @param file 
 * @return int 0 on success, -1 and sets errno if file is already open.
 */
int fileOpen(int flags, File *file)
{
    if(file->flags & O_OPEN)
    {
        errno = EINVAL;
        return -1;
    }
    file->flags = O_OPEN | flags | (file->flags & O_LOCK);
    return 0;
}

/**
 * @brief Closes the file, clearing its flags.
 * 
 * @param file
 * @returns int 0 on success, -1 and sets errno if file is not open.
 */
int fileClose(File *file)
{
    if(!(file->flags & O_OPEN))
    {
        errno = EINVAL;
        return -1;
    }
    file->flags = file->flags & O_LOCK;
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

