#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "COMMON/message.h"
#include "COMMON/macros.h"

//#define DEBUG

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
int messageInit(size_t size, void *content, const char *info, int type, int status, Message *m)
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
    size_t infoSize = 0;
#ifdef DEBUG
    printf("SEND:start:%d\n", fd);
#endif
    if (m->info != NULL)
    {
        infoSize = strlen(m->info) + 1;
    }
    else
    {
        infoSize = 0;
    }
    SAFE_ERROR_CHECK(writeWrapper(fd, &m->type, sizeof(int)));
    SAFE_ERROR_CHECK(writeWrapper(fd, &m->status, sizeof(int)));

    SAFE_ERROR_CHECK(writeWrapper(fd, &infoSize, sizeof(size_t)));
    if (infoSize != 0)
    {
        SAFE_ERROR_CHECK(writeWrapper(fd, m->info, infoSize));
    }
#ifdef DEBUG
    printf("SEND:infoSize:%ld status:%d type:%d fd:%d\n", infoSize, m->status, m->type, fd);
#endif
    SAFE_ERROR_CHECK(writeWrapper(fd, &m->size, sizeof(size_t)));
#ifdef DEBUG
    printf("SEND:size:%ld fd:%d\n", m->size, fd);
#endif
    if (m->size != 0)
    {
        SAFE_ERROR_CHECK(writeWrapper(fd, m->content, m->size));
    }

#ifdef DEBUG
    printf("SEND:done:%d\n", fd);
#endif

    return 0;
}

int receiveMessage(int fd, Message *m)
{
    size_t infoSize = 0;

#ifdef DEBUG
    printf("REC:start:%d\n", fd);
#endif

    SAFE_ERROR_CHECK(readWrapper(fd, &m->type, sizeof(int)));
    SAFE_ERROR_CHECK(readWrapper(fd, &m->status, sizeof(int)));

    SAFE_ERROR_CHECK(readWrapper(fd, &infoSize, sizeof(size_t)));

#ifdef DEBUG
    printf("REC:infoSize:%ld status:%d type:%d fd:%d\n", infoSize, m->status, m->type, fd);
#endif
    if (infoSize == 0)
    {
        m->info = NULL;
    }
    else
    {
        SAFE_NULL_CHECK(m->info = malloc(infoSize));
        SAFE_ERROR_CHECK(readWrapper(fd, m->info, infoSize));
    }

    SAFE_ERROR_CHECK(readWrapper(fd, &m->size, sizeof(size_t)));
#ifdef DEBUG
    printf("REC:size:%ld fd:%d\n", m->size, fd);
#endif
    if (m->size <= 0)
    {
        m->content = NULL;
#ifdef DEBUG
        puts("REC:content nulled!");
#endif
    }
    else
    {

        SAFE_NULL_CHECK(m->content = malloc(m->size));
        SAFE_ERROR_CHECK(readWrapper(fd, m->content, m->size));
    }
#ifdef DEBUG
    printf("REC:done:%d\n", fd);
#endif
    return 0;
}
/**
 * @brief Will read count bytes into the buffer, or return an error.
 *
 * @param fd the filedescriptor from which to read.
 * @param buf the buffer in which to read.
 * @param count how many bytes to read.
 * @return int 0 on success, -1 on error.
 */
int readWrapper(int fd, void *buf, size_t count)
{
    ssize_t curLoc = 0;
    ssize_t tmp;
    while (curLoc < count)
    {
        tmp = read(fd, buf + curLoc, count - curLoc);
        if (tmp <= 0)
        {
            printf("Error when reading; err:%ld, Expected to read:%ld, Actually read:%ld\n", tmp, count, curLoc);
            return -1;
        }
        curLoc += tmp;
    }

    return 0;
}

/**
 * @brief Will write count bytes into the buffer, or return an error.
 *
 * @param fd the filedescriptor in which to write.
 * @param buf the buffer from which to read.
 * @param count how many bytes to write.
 * @return int 0 on success, -1 on error.
 */
int writeWrapper(int fd, void *buf, size_t count)
{
    ssize_t curLoc = 0;
    ssize_t tmp;
    while (curLoc < count)
    {
        tmp = write(fd, buf + curLoc, count - curLoc);
        if (tmp <= 0)
        {
            printf("Error when writing; err:%ld, Expected to write:%ld, Actually wrote:%ld\n", tmp, count, curLoc);
            return -1;
        }
        curLoc += tmp;
    }

    return 0;
}