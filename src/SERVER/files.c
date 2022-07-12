#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include <zlib.h>
#include "SERVER/files.h"
#include "COMMON/macros.h"
#include "COMMON/logging.h"

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
    file->compressedSize = 0;
    file->isCompressed = 0;
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
    free((void *)file->name);
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
    printf("File writing with size:%ld\n", size);

    file->size = size;

    if (size != 0)
    {
        UNSAFE_NULL_CHECK(file->content = realloc(file->content, size));

        memcpy(file->content, content, size);
    }
    else
    {
        free(file->content);
        file->content = NULL;
    }

    puts("And memcpy done..");
    if (file->isCompressed)
    {
        file->isCompressed = 0;
        return fileCompress(file);
    }

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
    int shouldCompress = file->isCompressed;
    if (file->isCompressed)
    {
        SAFE_ERROR_CHECK(fileDecompress(file));
    }

    UNSAFE_NULL_CHECK(file->content = realloc(file->content, size + file->size));
    memcpy(file->content + file->size, content, size);

    file->size += size;

    if (shouldCompress)
    {
        return fileCompress(file);
    }
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
    if (file->isCompressed)
    {
        return uncompress(buf, &bufsize, file->content, file->compressedSize);
    }

    if (bufsize < file->size)
        n = bufsize;
    else
        n = file->size;

    memcpy(buf, file->content, n);

    return 0;
}

/**
 * @brief Trys to lock the file, without blocking.
 *
 * @param file
 * @return int 0 if succesfull, -1 and sets errno if file is already locked.
 */
int fileTryLock(File *file)
{
    int res;

    res = pthread_mutex_trylock(file->mtx);
    if (res != 0)
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
 * @brief Compresses the files contents.
 *
 * @param file the file to be compressed.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileCompress(File *file)
{
    void *buf;
    char log[500];

    puts("File compressing");
    if (file->isCompressed)
    {
        puts("Already compressed smh");
        errno = EINVAL;
        return -1;
    }

    file->compressedSize = compressBound(file->size);
    SAFE_NULL_CHECK(buf = malloc(file->compressedSize));
    if (compress(buf, &(file->compressedSize), file->content, file->size) != Z_OK)
    {
        perror("Call to compress failed");
        free(buf);
        return -1;
    }
    free(file->content);
    file->content = buf;
    SAFE_NULL_CHECK(file->content = realloc(file->content, file->compressedSize));
    file->isCompressed = 1;

    sprintf(log, "Compressed file %s from size:%ld to size:%ld", file->name, file->size, file->compressedSize);
    logger(log);

    return 0;
}

/**
 * @brief Decompresses the files contents.
 *
 * @param file the file to decompress.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileDecompress(File *file)
{
    void *buf;
    char log[500];

    if (!file->isCompressed)
    {
        errno = EINVAL;
        return -1;
    }
    SAFE_NULL_CHECK(buf = malloc(file->size));
    if (uncompress(buf, &(file->size), file->content, file->compressedSize) != Z_OK)
    {
        perror("Call to uncompress failed");
        free(buf);
        return -1;
    }
    free(file->content);
    file->content = buf;
    SAFE_NULL_CHECK(file->content = realloc(file->content, file->size));
    file->isCompressed = 0;

    sprintf(log, "Decompressed file %s from size:%ld to size:%ld", file->name, file->compressedSize, file->size);
    logger(log);

    file->compressedSize = 0;

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
 * @brief Get the File's contents true memory occupation. If file is not compressed then this is equal to getFileSize, otherwise it will return the commpressed size of the file's contents.
 *
 * @param file
 * @return uint64_t
 */
uint64_t getFileTrueSize(File *file)
{
    if (file->isCompressed)
        return file->compressedSize;
    else
        return file->size;
}

/**
 * @brief Get the File's name.
 *
 * @param file
 * @return char*
 */
const char *getFileName(File *file)
{
    return file->name;
}
