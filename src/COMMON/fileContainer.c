#include <stdlib.h>
#include <string.h>

#include "COMMON/fileContainer.h"
#include "COMMON/macros.h"

/**
 * @brief Initializes the given container, with given parameters.
 *
 * @param size the size of the new containers content.
 * @param content the content of the container, size bytes will be copyed from here inside of the containers content.
 * @param name the name of the container.
 * @param fc a pointer to the container to be initialized.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int containerInit(size_t size, void *content, const char *name, FileContainer *fc)
{
    fc->size = size;
    if (size != 0)
    {
        SAFE_NULL_CHECK(fc->content = malloc(size));
        memcpy(fc->content, content, size);
    }
    else
    {
        fc->content = NULL;
    }
    SAFE_NULL_CHECK(fc->name = malloc(strlen(name) + 1));
    strcpy(fc->name, name);
    return 0;
}

/**
 * @brief Destroys given container, freeing it's memory.
 *
 * @param fc the container to be destroyed.
 */
void destroyContainer(FileContainer *fc)
{
    if (fc->size != 0)
    {
        free(fc->content);
    }
    fc->content = NULL;
    free(fc->name);
    fc->name = NULL;
    fc->size = 0;
}

/**
 * @brief Serializes the given container, mallocing a new buffer and returning it.
 *
 * @param fc the fileContainer to serialize.
 * @param buf where the serialized container will be stored.
 * @param size where the size of the new buf will be stored.
 * @return int 0 on success, -1 and sets errno on failure
 */
int serializeContainer(FileContainer fc, void **buf, size_t *size)
{
    size_t cur = 0;
    size_t nameLen = strlen(fc.name) + 1;

    *size = calcSize(fc);

    SAFE_NULL_CHECK(*buf = malloc(*size));

    memcpy(*buf + cur, (void *)&fc.size, sizeof(size_t));
    cur += sizeof(size_t);
    memcpy(*buf + cur, (void *)fc.name, nameLen);
    cur += nameLen;
    if (fc.size > 0)
    {
        memcpy(*buf + cur, fc.content, fc.size);
    }
    return 0;
}

/**
 * @brief Deserializes the given buffer, returning a new initialized FileContainer.
 *
 * @param buf the buffer containing a serialized FileContainer.
 * @param size the size of buf.
 * @return FileContainer a new initialized FileContainer.
 */
FileContainer deserializeContainer(void *buf, size_t size)
{
    FileContainer fc;
    size_t cur = 0;
    size_t nameLen = -1;

    fc.size = *(size_t *)(buf + cur);
    cur += sizeof(size_t);

    nameLen = strlen((char *)(buf + cur)) + 1;

    UNSAFE_NULL_CHECK(fc.name = malloc(nameLen * sizeof(char)));

    memcpy(fc.name, buf + cur, nameLen);
    cur += nameLen;

    if (fc.size > 0)
    {
        UNSAFE_NULL_CHECK(fc.content = malloc(fc.size));
        memcpy(fc.content, buf + cur, fc.size);
    }
    else
    {
        fc.content = NULL;
    }

    return fc;
}

/**
 * @brief Serializes an array of n FileContainers, mallocing a new buffer and returning it in buf.
 *
 * @param fc the array to be serialized.
 * @param n the length of fc.
 * @param size where the size of buf will be stored.
 * @param buf where the serialized array will be stored.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int serializeContainerArray(FileContainer *fc, size_t n, size_t *size, void **buf)
{
    int i;
    void *curBuf = NULL;
    size_t curSize, finalSize = 0, curLoc = 0;

    if (n == 0)
    {
        puts("Called serialize with 0 size array, this shouldn't happen!");
        return -1;
    }
    for (i = 0; i < n; i++)
    {
        finalSize += sizeof(size_t) + calcSize(fc[i]);
    }

    *size = finalSize;
    SAFE_NULL_CHECK(*buf = malloc(finalSize));

    for (i = 0; i < n; i++)
    {
        SAFE_ERROR_CHECK(serializeContainer(fc[i], &curBuf, &curSize))

        memcpy(*buf + curLoc, (void *)&curSize, sizeof(size_t));
        curLoc += sizeof(size_t);
        memcpy(*buf + curLoc, curBuf, curSize);
        curLoc += curSize;
        free(curBuf);
    }
    return 0;
}

/**
 * @brief Deserializes given buffer as an array of FileContainers.
 *
 * @param buf the buffer to be deserialized.
 * @param size the size of the given buffer.
 * @param n where the size of the array will be stored.
 * @return FileContainer* the deserialized array.
 */
FileContainer *deserializeContainerArray(void *buf, size_t size, size_t *n)
{
    FileContainer *result;
    size_t curLoc = 0, curSize = 0, count = 0, i;

    while (curLoc < size)
    {
        curSize = *(size_t *)(buf + curLoc);
        curLoc += sizeof(size_t) + curSize;
        count++;
    }

    UNSAFE_NULL_CHECK(result = malloc(sizeof(FileContainer) * count));

    curLoc = 0;
    for (i = 0; i < count; i++)
    {
        curSize = *(size_t *)(buf + curLoc);
        curLoc += sizeof(size_t);
        result[i] = deserializeContainer(buf + curLoc, curSize);
        curLoc += curSize;
    }

    *n = count;
    return result;
}

/**
 * @brief Returns the serialized size of given FileContainer, in bytes.
 *
 * @param fc the FileContainer to serialize.
 * @return size_t the size of fc, once serialized.
 */
size_t calcSize(FileContainer fc)
{
    return fc.size + strlen(fc.name) + 1 + sizeof(size_t);
}
