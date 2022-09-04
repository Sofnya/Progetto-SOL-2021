#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include <zlib.h>
#include "SERVER/files.h"
#include "COMMON/macros.h"
#include "SERVER/logging.h"

/**
 * @brief Initializes given file with the correct name.
 *
 * @param name the pathname of the file
 * @param file the file to initialize
 * @return int 0 on success, -1 and sets errno otherwise
 */
int fileInit(const char *name, int isCompressed, File *file)
{
    size_t nameSize;

    nameSize = strlen(name) + 1;
    SAFE_NULL_CHECK(file->name = malloc(sizeof(char) * nameSize));
    strcpy(file->name, name);

    file->size = 0;
    file->compressedSize = 0;
    file->isCompressed = isCompressed;
    file->content = NULL;

    file->isDestroyed = 0;
    file->isLocked = 0;
    file->waitingThreads = 0;

    SAFE_NULL_CHECK(file->mtx = malloc(sizeof(pthread_mutex_t)));
    SAFE_NULL_CHECK(file->waitingLock = malloc(sizeof(pthread_mutex_t)));
    SAFE_NULL_CHECK(file->wake = malloc(sizeof(pthread_cond_t)));
    PTHREAD_CHECK(pthread_mutex_init(file->mtx, NULL));
    PTHREAD_CHECK(pthread_mutex_init(file->waitingLock, NULL));
    PTHREAD_CHECK(pthread_cond_init(file->wake, NULL));

    SAFE_NULL_CHECK(file->metadata = malloc(sizeof(Metadata)));
    metadataInit(name, file->metadata);

    return 0;
}

/**
 * @brief Destroys given file, freeing it's resources.
 *
 * @param file the file to be destroyed
 */
void fileDestroy(File *file)
{
    PTHREAD_CHECK(pthread_mutex_lock(file->waitingLock));
    file->isDestroyed = 1;
    while (file->waitingThreads > 0)
    {
        PTHREAD_CHECK(pthread_cond_broadcast(file->wake));
        PTHREAD_CHECK(pthread_cond_wait(file->wake, file->waitingLock));
    }
    PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));
    free((void *)file->name);
    free(file->content);

    PTHREAD_CHECK(pthread_mutex_unlock(file->mtx));
    PTHREAD_CHECK(pthread_mutex_destroy(file->mtx));
    PTHREAD_CHECK(pthread_mutex_destroy(file->waitingLock));
    PTHREAD_CHECK(pthread_cond_destroy(file->wake));
    free(file->mtx);
    free(file->waitingLock);
    free(file->wake);

    metadataDestroy(file->metadata);
    free(file->metadata);
}

/**
 * @brief Initializes given metadata, should be done at file creation.
 *
 * @param metadata the Metadata to be initialized.
 * @return int 0 on success, -1 on failure.
 */
int metadataInit(const char *name, Metadata *metadata)
{
    metadata->creationTime = time(NULL);
    metadata->lastAccess = time(NULL);
    metadata->numberAccesses = 0;
    metadata->size = 0;

    SAFE_NULL_CHECK(metadata->metadataLock = malloc(sizeof(pthread_mutex_t)));
    PTHREAD_CHECK(pthread_mutex_init(metadata->metadataLock, NULL));

    SAFE_NULL_CHECK(metadata->name = malloc((strlen(name) + 1) * sizeof(char)));
    strcpy(metadata->name, name);

    return 0;
}

/**
 * @brief Destroys given metadata, freeing it's resources.
 *
 * @param metadata the Metadata to be destroyed.
 */
void metadataDestroy(Metadata *metadata)
{
    PTHREAD_CHECK(pthread_mutex_destroy(metadata->metadataLock));
    free(metadata->metadataLock);
    free(metadata->name);
}

/**
 * @brief Atomically updates metadata to represent a fileAccess.
 *
 * @param metadata the metadata to be updated.
 * @return int 0 on success, -1 on failure.
 */
int metadataAccess(Metadata *metadata)
{
    PTHREAD_CHECK(pthread_mutex_lock(metadata->metadataLock));

    metadata->numberAccesses++;
    metadata->lastAccess = time(NULL);

    PTHREAD_CHECK(pthread_mutex_unlock(metadata->metadataLock));

    return 0;
}

/**
 * @brief Atomically updates metadata with the new size.
 *
 * @param metadata the metadata to be updated.
 * @param size  the new size.
 * @return int 0 on success, -1 on failure.
 */
int metadataUpdateSize(Metadata *metadata, size_t size)
{
    PTHREAD_CHECK(pthread_mutex_lock(metadata->metadataLock));

    metadata->size = size;

    PTHREAD_CHECK(pthread_mutex_unlock(metadata->metadataLock));

    return 0;
}

/**
 * @brief Writes the given content to the file.
 *
 * @param content the content to be written.
 * @param size the size of content.
 * @param file the file to be modified.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileWrite(const void *content, size_t size, File *file)
{
    file->size = size;
    metadataAccess(file->metadata);
    metadataUpdateSize(file->metadata, size);
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
int fileAppend(const void *content, size_t size, File *file)
{
    int shouldCompress = file->isCompressed;

    metadataAccess(file->metadata);

    if (file->isCompressed)
    {
        SAFE_ERROR_CHECK(fileDecompress(file));
    }

    UNSAFE_NULL_CHECK(file->content = realloc(file->content, size + file->size));
    memcpy(file->content + file->size, content, size);

    file->size += size;

    metadataUpdateSize(file->metadata, file->size);

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
int fileRead(void *buf, size_t bufsize, File *file)
{
    size_t n;

    metadataAccess(file->metadata);
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
    metadataAccess(file->metadata);

    PTHREAD_CHECK(pthread_mutex_lock(file->waitingLock));

    res = pthread_mutex_trylock(file->mtx);
    if (res == 0)
    {
        file->isLocked = 1;
    }

    PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));
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
    metadataAccess(file->metadata);

    PTHREAD_CHECK(pthread_mutex_lock(file->waitingLock));

    file->waitingThreads++;
    while (!file->isDestroyed && file->isLocked)
    {
        PTHREAD_CHECK(pthread_cond_wait(file->wake, file->waitingLock));
    }
    if (file->isDestroyed)
    {
        file->waitingThreads--;
        PTHREAD_CHECK(pthread_cond_broadcast(file->wake));
        PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));
        return -1;
    }

    PTHREAD_CHECK(pthread_mutex_lock(file->mtx));
    file->isLocked = 1;
    file->waitingThreads--;

    PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));

    return 0;
}

/**
 * @brief Unlocks the file.
 *
 * @param file the file to be unlocked.
 * @return int 0 on success, -1 on failure.
 */
int fileUnlock(File *file)
{
    metadataAccess(file->metadata);

    PTHREAD_CHECK(pthread_mutex_lock(file->waitingLock));
    PTHREAD_CHECK(pthread_cond_broadcast(file->wake));
    PTHREAD_CHECK(pthread_mutex_unlock(file->mtx));

    file->isLocked = 0;
    PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));
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
    char *log;

    if (file->isCompressed)
    {
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

    SAFE_NULL_CHECK(log = malloc(500 + strlen(file->name)));
    sprintf(log, ">From:%ld >To:%ld >Name:%s ", file->size, file->compressedSize, file->name);
    logger(log, "COMPRESSED");

    free(log);

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
    char *log;

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

    SAFE_NULL_CHECK(log = malloc(strlen(file->name) + 500));
    sprintf(log, ">From:%ld >To:%ld >Name:%s ", file->compressedSize, file->size, file->name);
    logger(log, "DECOMPRESSED");
    free(log);

    file->compressedSize = 0;

    return 0;
}

/**
 * @brief Get the File's size.
 *
 * @param file the file to query.
 * @return size_t the uncompressed size of the file.
 */
size_t getFileSize(File *file)
{
    return file->size;
}

/**
 * @brief Get the File's contents true memory occupation. If file is not compressed then this is equal to getFileSize, otherwise it will return the commpressed size of the file's contents.
 *
 * @param file the file to query.
 * @return size_t the true current size of the file.
 */
size_t getFileTrueSize(File *file)
{
    if (file->isCompressed)
        return file->compressedSize;
    else
        return file->size;
}

/**
 * @brief Get the File's name.
 *
 * @param file the file to query.
 * @return char* the name of the file
 */
const char *getFileName(File *file)
{
    return file->name;
}
