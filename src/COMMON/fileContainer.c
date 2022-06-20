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
int containerInit(uint64_t size, void *content, char *name, FileContainer *fc)
{
    fc->size = size;
    SAFE_NULL_CHECK(fc->content = malloc(size));
    memcpy(content, fc->content, size);
    
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
    free(fc->content);
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
int serializeContainer(FileContainer fc, void **buf, uint64_t *size)
{
    uint64_t cur = 0;
    size_t nameLen = strlen(fc.name) + 1;

    *size  = fc.size + nameLen + sizeof(uint64_t);

    SAFE_NULL_CHECK(*buf = malloc(*size));

    memcpy(*buf + cur, (void *)&fc.size, sizeof(uint64_t));
    cur += sizeof(uint64_t);
    memcpy(*buf + cur, (void *)fc.name, nameLen);
    cur += nameLen;
    memcpy(*buf + cur, fc.content, fc.size);
    
    return 0;
}


/**
 * @brief Deserializes the given buffer, returning a new initialized FileContainer.
 * 
 * @param buf the buffer containing a serialized FileContainer.
 * @param size the size of buf.
 * @return FileContainer a new initialized FileContainer.
 */
FileContainer deserializeContainer(void *buf, uint64_t size)
{
    FileContainer fc;
    uint64_t cur = 0;
    size_t nameLen = -1;

    fc.size = (uint64_t) (buf + cur);
    cur += sizeof(uint64_t);
    
    nameLen = strlen((char *)(buf + cur)) + 1;

    UNSAFE_NULL_CHECK(fc.name = malloc(nameLen * sizeof(char)));

    memcpy(fc.name, buf + cur, nameLen);
    cur += nameLen;

    UNSAFE_NULL_CHECK(fc.content = malloc(fc.size));
    memcpy(fc.content, buf + cur, fc.size);

    return fc;
}