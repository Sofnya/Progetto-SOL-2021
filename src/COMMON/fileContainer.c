#include <stdlib.h>
#include <string.h>

#include "COMMON/fileContainer.h"
#include "COMMON/macros.h"

/**
 * @brief Initializes given FileContainer, with given parameters.
 *
 * @param size the size of the new FileContainer's content.
 * @param content the content of the FileContainer, size bytes will be copyed from here inside of the FileContainer's content.
 * @param name the name of the FileContainer.
 * @param fc a pointer to the FileContainer to initialize.
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
 * @brief Destroys given FileContainer, freeing it's memory.
 *
 * @param fc the FileContainer to destroy.
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
 * @brief Serializes given FileContainer, allocating a new buffer and returning it in buf.
 *
 * @param fc the FileContainer to serialize.
 * @param buf where the serialized FileContainer will be stored.
 * @param size where the size of the new buf will be stored.
 * @return int 0 on success, -1 and sets errno on failure
 */
int serializeContainer(FileContainer fc, void **buf, size_t *size)
{
    size_t cur = 0;
    size_t nameLen = strlen(fc.name) + 1;

    // First calculate the size and allocate an appropriate buffer.
    *size = calcSize(fc);
    SAFE_NULL_CHECK(*buf = malloc(*size));

    // We then serialize the FileContainer inside of the buffer buf in the form:
    // [sizeof(size_t)]ContentSize [nameLen]Name (it's null terminated so we don't need to know it's length.) [ContentSize]Content.
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
 * @brief Deserializes given buffer, returning a new initialized FileContainer.
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

    // First we recover the size from buf.
    fc.size = *(size_t *)(buf + cur);
    cur += sizeof(size_t);

    // Then the length of name, which was a null-terminated string.
    nameLen = strlen((char *)(buf + cur)) + 1;
    // And copy the name to fc.name.
    UNSAFE_NULL_CHECK(fc.name = malloc(nameLen * sizeof(char)));
    memcpy(fc.name, buf + cur, nameLen);
    cur += nameLen;

    // Finally, if a content was present, we copy it in fc.content.
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
 * @brief Serializes an array of n FileContainers, allocating a new buffer and returning it in buf.
 *
 * @param fc the array to serialize.
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

    // First we calculate the size of the final buffer, and allocate it.
    for (i = 0; i < n; i++)
    {
        finalSize += sizeof(size_t) + calcSize(fc[i]);
    }

    *size = finalSize;
    SAFE_NULL_CHECK(*buf = malloc(finalSize));

    // We then serialize the array of containers in the form: [sizeof(size_t)]Serialized size [size bytes]Serialized Container etc.
    // We just keep track of how large the next serialized FileContainer by putting it's size right before it, so that we can know how many bytes to deserialize afterwards.
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
 * @param buf the buffer to deserialize.
 * @param size the size of the given buffer.
 * @param n where the size of the array will be stored.
 * @return FileContainer* the deserialized array.
 */
FileContainer *deserializeContainerArray(void *buf, size_t size, size_t *n)
{
    FileContainer *result;
    size_t curLoc = 0, curSize = 0, count = 0, i;

    // First we count how many FileContainers are present in the array.
    while (curLoc < size)
    {
        curSize = *(size_t *)(buf + curLoc);
        curLoc += sizeof(size_t) + curSize;
        count++;
    }

    // Once we know we can allocate our result array.
    UNSAFE_NULL_CHECK(result = malloc(sizeof(FileContainer) * count));

    // And then do the actual deserialization.
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
