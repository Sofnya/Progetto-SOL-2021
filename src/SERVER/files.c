#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include <zlib.h>
#include "SERVER/files.h"
#include "COMMON/macros.h"
#include "SERVER/logging.h"

/**
 * @brief Initializes given File with given name.
 *
 * @param name the pathname of the File
 * @param file the File to initialize
 * @param isCompressed should be 0 if the File is not to be compressed, 1 otherwise.
 * @return int 0 on success, -1 and sets errno otherwise
 */
int fileInit(const char *name, int isCompressed, File *file)
{
    size_t nameSize;

    // Every File has it's own copy of the name, to avoid having to collaborate with other data structures in freeing it afterwars.
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
 * @brief Destroys given File, freeing it's resources.
 *
 * @param file the File to be destroyed
 * @return int 0 on success, -1 on failure.
 */
int fileDestroy(File *file)
{
    // We have to do this to avoid destroying the File someone is waiting for a lock on.

    // First enter the waitingLock critical zone.
    PTHREAD_CHECK(pthread_mutex_lock(file->waitingLock));
    // Tell all threads the File is getting destroyed.
    file->isDestroyed = 1;

    // And wake them up, they will terminate unsuccesfully.
    while (file->waitingThreads > 0)
    {
        PTHREAD_CHECK(pthread_cond_broadcast(file->wake));
        PTHREAD_CHECK(pthread_cond_wait(file->wake, file->waitingLock));
    }
    PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));

    // Now we are guaranteed to be the only thread with the File, as this File is no longer accessible from the FileSystem.
    // As such we can finally destroy it safely.

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

    return 0;
}

/**
 * @brief Initializes given Metadata, should be done at File creation.
 *
 * @param metadata the Metadata to be initialized.
 * @return int 0 on success, -1 on failure.
 */
int metadataInit(const char *name, Metadata *metadata)
{
    metadata->creationTime = time(NULL);

    // We count creation as an access.
    metadata->lastAccess = time(NULL);
    metadata->numberAccesses = 1;

    metadata->size = 0;

    SAFE_NULL_CHECK(metadata->metadataLock = malloc(sizeof(pthread_mutex_t)));
    PTHREAD_CHECK(pthread_mutex_init(metadata->metadataLock, NULL));

    SAFE_NULL_CHECK(metadata->name = malloc((strlen(name) + 1) * sizeof(char)));
    strcpy(metadata->name, name);

    return 0;
}

/**
 * @brief Destroys given Metadata, freeing it's resources.
 *
 * @param metadata the Metadata to be destroyed.
 */
void metadataDestroy(Metadata *metadata)
{
    VOID_PTHREAD_CHECK(pthread_mutex_destroy(metadata->metadataLock));
    free(metadata->metadataLock);
    free(metadata->name);
}

/**
 * @brief Atomically updates Metadata to represent a fileAccess.
 *
 * @param metadata the Metadata to be updated.
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
 * @brief Atomically updates Metadata with its new size.
 *
 * @param metadata the Metadata to be updated.
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
 * @brief Writes given content to given File.
 *
 * @param content the content to be written.
 * @param size the size of content.
 * @param file the File to modify.
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

    // If the File is compressed, we need to recompress it at the end.
    if (file->isCompressed)
    {
        file->isCompressed = 0;
        return fileCompress(file);
    }

    return 0;
}

/**
 * @brief Appends the given content at the end of the File.
 *
 * @param content the content to be written.
 * @param size the size of the new content.
 * @param file the File to modify.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileAppend(const void *content, size_t size, File *file)
{
    // Need to remember if the File was compressed at the start.
    int shouldCompress = file->isCompressed;

    metadataAccess(file->metadata);

    // To be able to work on it we first need to decompress it.
    if (file->isCompressed)
    {
        SAFE_ERROR_CHECK(fileDecompress(file));
    }

    UNSAFE_NULL_CHECK(file->content = realloc(file->content, size + file->size));
    memcpy(file->content + file->size, content, size);

    file->size += size;

    metadataUpdateSize(file->metadata, file->size);

    // If it was compressed at the start, we compress it again now.
    if (shouldCompress)
    {
        return fileCompress(file);
    }
    return 0;
}

/**
 * @brief Read up to bufsize bytes from the Files contents inside of the given buf.
 *
 * @param buf where the contents will be returned.
 * @param bufsize the size of buf.
 * @param file the File to read.
 * @return int 0 on success, -1 and sets errno otherwise.
 */
int fileRead(void *buf, size_t bufsize, File *file)
{
    size_t n;

    metadataAccess(file->metadata);

    // If File is compressed, we directly decompress it inside of the buffer.
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
 * @brief Trys to lock given File, without blocking.
 *
 * @param file the File to lock.
 * @return int 0 if succesfull, -1 and sets errno if File is already locked.
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
 * @brief Locks given File.
 *
 * @param file the File to lock.
 * @return int 0 on success, -1 and sets ERRNO on failure.
 */
int fileLock(File *file)
{
    metadataAccess(file->metadata);

    // We need to be careful that the File we are waiting to lock has not been destroyed in the meantime.
    PTHREAD_CHECK(pthread_mutex_lock(file->waitingLock));

    // We notify others of our presence.
    file->waitingThreads++;

    // While the File remains locked, and not destroyed, we wait for something to change.
    while (!file->isDestroyed && file->isLocked)
    {
        PTHREAD_CHECK(pthread_cond_wait(file->wake, file->waitingLock));
    }
    // If the File has been scheduled to be destroyed, we terminate unsuccesfully.
    if (file->isDestroyed)
    {
        file->waitingThreads--;

        // And wake all other threads for safety.
        PTHREAD_CHECK(pthread_cond_broadcast(file->wake));
        PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));

        errno = ECANCELED;
        return -1;
    }

    // Otherwise just lock the File lol.
    PTHREAD_CHECK(pthread_mutex_lock(file->mtx));
    file->isLocked = 1;
    file->waitingThreads--;

    PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));

    return 0;
}

/**
 * @brief Unlocks given File.
 *
 * @param file the File to be unlocked.
 * @return int 0 on success, -1 on failure.
 */
int fileUnlock(File *file)
{
    metadataAccess(file->metadata);

    PTHREAD_CHECK(pthread_mutex_lock(file->waitingLock));

    // Remember to notify all threads waiting for a lock.
    PTHREAD_CHECK(pthread_cond_broadcast(file->wake));
    PTHREAD_CHECK(pthread_mutex_unlock(file->mtx));

    file->isLocked = 0;
    PTHREAD_CHECK(pthread_mutex_unlock(file->waitingLock));
    return 0;
}

/**
 * @brief Compresses given File contents.
 *
 * @param file the File to compress.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileCompress(File *file)
{
    void *buf;
    char *log;

    // File shouldn't be compressed twice.
    if (file->isCompressed)
    {
        errno = EINVAL;
        return -1;
    }

    // First get a rough bound for our final compressed size.
    file->compressedSize = compressBound(file->size);
    SAFE_NULL_CHECK(buf = malloc(file->compressedSize));

    // Call zlib.
    if (compress(buf, &(file->compressedSize), file->content, file->size) != Z_OK)
    {
        perror("Call to compress failed");
        free(buf);
        return -1;
    }
    free(file->content);
    file->content = buf;

    // Realloc our File's contents to it's actual compressed size, saving space.
    SAFE_NULL_CHECK(file->content = realloc(file->content, file->compressedSize));
    file->isCompressed = 1;

    SAFE_NULL_CHECK(log = malloc(500 + strlen(file->name)));
    sprintf(log, ">From:%ld >To:%ld >Name:%s ", file->size, file->compressedSize, file->name);
    logger(log, "COMPRESSED");

    free(log);

    return 0;
}

/**
 * @brief Decompresses given Files contents.
 *
 * @param file the File to decompress.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int fileDecompress(File *file)
{
    void *buf;
    char *log;

    // File should be compressed lol.
    if (!file->isCompressed)
    {
        errno = EINVAL;
        return -1;
    }

    // We keep track of the File's uncompressed size in file->size.
    SAFE_NULL_CHECK(buf = malloc(file->size));

    // Call zlib.
    if (uncompress(buf, &(file->size), file->content, file->compressedSize) != Z_OK)
    {
        perror("Call to uncompress failed");
        free(buf);
        return -1;
    }
    free(file->content);
    file->content = buf;

    // Just to be safe, realloc to the acutal File size.
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
 * @param file the File to query.
 * @return size_t the uncompressed size of the File.
 */
size_t getFileSize(File *file)
{
    return file->size;
}

/**
 * @brief Get the File's contents true memory occupation. If File is not compressed then this is equal to getFileSize, otherwise it will return the commpressed size of the File's contents.
 *
 * @param file the File to query.
 * @return size_t the true current size of the File.
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
 * @param file the File to query.
 * @return char* the name of the File
 */
const char *getFileName(File *file)
{
    return file->name;
}
