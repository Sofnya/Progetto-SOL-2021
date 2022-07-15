#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "COMMON/message.h"
#include "COMMON/macros.h"

/**
 * @brief Initializes a message with given size, content and message.
 *
 * @param size the size of content.
 * @param content the content of the message.
 * @param info a info string.
 * @param type specifies the type of message that is being sent.
 * @param status the status code of the message.
 * @param m the message to be initialized.
 * @return int 0 on a success, -1 and sets errno on error.
 */
int messageInit(uint64_t size, void *content, const char *info, int type, int status, Message *m)
{
    size_t len;
    if (size < 0 || m == NULL)
    {
        errno = EINVAL;
        return -1;
    }

    m->size = size;
    if (size != 0)
    {
#ifdef DEBUG
        printf("Init: Allocating %ld bytes for content\n", size);
#endif

        SAFE_NULL_CHECK(m->content = malloc(size));
        memcpy(m->content, content, size);
    }
    else
    {
        m->content = NULL;
    }

    if (info != NULL)
    {
        len = strlen(info);

#ifdef DEBUG
        printf("Init: Allocating %ld bytes for info\n", len + 1);
#endif

        SAFE_NULL_CHECK(m->info = malloc(len + 1));
        strcpy(m->info, info);
        m->info[len] = '\00';
    }
    else
    {
        m->info = NULL;
    }

    m->type = type;
    m->status = status;

    return 0;
}

/**
 * @brief Destroyes given message, freeing it's memory.
 *
 * @param m the message to be destroyed.
 */
void messageDestroy(Message *m)
{
    if (m->content != NULL)
    {
        free(m->content);
    }
    if (m->info != NULL)
    {
        free(m->info);
    }
}

/**
 * @brief Sends given message on socket fd.
 *
 * @param fd
 * @param m
 * @return int 0 on success, -1 and sets errno on failure.
 */
int sendMessage(int fd, Message *m)
{
    uint64_t infoSize;

    if (m->info != NULL)
    {
        infoSize = strlen(m->info) + 1;
    }
    else
    {
        infoSize = 0;
    }
    SAFE_ERROR_CHECK(write(fd, &m->type, sizeof(int)));
    SAFE_ERROR_CHECK(write(fd, &m->status, sizeof(int)));

    SAFE_ERROR_CHECK(write(fd, &infoSize, sizeof(uint64_t)));
    if (infoSize != 0)
    {
        SAFE_ERROR_CHECK(write(fd, m->info, infoSize));
    }
    SAFE_ERROR_CHECK(write(fd, &m->size, sizeof(uint64_t)));
    if (m->size != 0)
    {
        SAFE_ERROR_CHECK(write(fd, m->content, m->size));
    }

    return 0;
}

int receiveMessage(int fd, Message *m)
{
    uint64_t infoSize = 0;

#ifdef DEBUG
    printf("Receiving message on %d\n", fd);
#endif

    READ_CHECK(read(fd, &m->type, sizeof(int)));
    READ_CHECK(read(fd, &m->status, sizeof(int)));

    READ_CHECK(read(fd, &infoSize, sizeof(uint64_t)));

    printf("Received message of type:%d, and status:%d and infoSize:%ld\n", m->type, m->status, infoSize);

    if (infoSize <= 0)
    {
        m->info = NULL;
    }
    else
    {
#ifdef DEBUG
        printf("Allocating %ld bytes for info\n", infoSize);
#endif

        SAFE_NULL_CHECK(m->info = malloc(infoSize));
        READ_CHECK(read(fd, m->info, infoSize));
    }

    READ_CHECK(read(fd, &m->size, sizeof(uint64_t)));
    if (m->size <= 0)
    {
        m->content = NULL;
    }
    else
    {
#ifdef DEBUG
        printf("Allocating %ld bytes for content\n", m->size);
#endif

        SAFE_NULL_CHECK(m->content = malloc(m->size));
        READ_CHECK(read(fd, m->content, m->size));
    }

    return 0;
}
