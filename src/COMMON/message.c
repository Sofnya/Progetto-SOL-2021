#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "COMMON/message.h"
#include "COMMON/macros.h"

/**
 * @brief Initializes given Message with given size, content and message.
 *
 * @param size the size of content.
 * @param content the content of the Message.
 * @param info a info string.
 * @param type specifies the type of Message that is being sent.
 * @param status the status code of the Message.
 * @param m the Message to initialize.
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
 * @brief Destroyes given Message, freeing it's resources.
 *
 * @param m the Message to destroy.
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
 * @brief Sends given Message on socket fd.
 *
 * @param fd the socket on which to send given Message.
 * @param m the Message to send.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int sendMessage(int fd, Message *m)
{
    size_t infoSize = 0;
#ifdef DEBUG
    printf("SEND:start:%d\n", fd);
#endif

    // We are basically serializing the Message.
    if (m->info != NULL)
    {
        infoSize = strlen(m->info) + 1;
    }
    else
    {
        infoSize = 0;
    }
    // We first send the Message's type and status, which are of a known size (sizeof(int)).
    SAFE_PIPE_CHECK(writeWrapper(fd, &m->type, sizeof(int)));
    SAFE_PIPE_CHECK(writeWrapper(fd, &m->status, sizeof(int)));

    // Then we send the size of info, so that the receiving party can wait until they read all of it.
    SAFE_PIPE_CHECK(writeWrapper(fd, &infoSize, sizeof(size_t)));
    if (infoSize != 0)
    {
        // And send info, if present.
        SAFE_PIPE_CHECK(writeWrapper(fd, m->info, infoSize));
    }
#ifdef DEBUG
    printf("SEND:infoSize:%ld status:%d type:%d fd:%d\n", infoSize, m->status, m->type, fd);
#endif
    // We do the same with the Message's content, first sending it's size.
    SAFE_PIPE_CHECK(writeWrapper(fd, &m->size, sizeof(size_t)));
#ifdef DEBUG
    printf("SEND:size:%ld fd:%d\n", m->size, fd);
#endif
    if (m->size != 0)
    {
        // And then the content, if any.
        SAFE_PIPE_CHECK(writeWrapper(fd, m->content, m->size));
    }

#ifdef DEBUG
    printf("SEND:done:%d\n", fd);
#endif

    return 0;
}

/**
 * @brief Recieves a Message on given socket.
 *
 * @param fd the socket on which to recieve.
 * @param m where the recieved Message will be stored.
 * @return int 0 on success, -1 on failure.
 */
int receiveMessage(int fd, Message *m)
{
    size_t infoSize = 0;
    m->content = NULL;
    m->info = NULL;
    m->size = 0;
    m->status = 0;
    m->type = 0;
#ifdef DEBUG
    printf("REC:start:%d\n", fd);
#endif

    // Like sending but in reverse.
    SAFE_PIPE_CHECK(readWrapper(fd, &m->type, sizeof(int)));
    SAFE_PIPE_CHECK(readWrapper(fd, &m->status, sizeof(int)));

    SAFE_PIPE_CHECK(readWrapper(fd, &infoSize, sizeof(size_t)));

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
        SAFE_PIPE_CHECK(readWrapper(fd, m->info, infoSize));
    }

    SAFE_PIPE_CHECK(readWrapper(fd, &m->size, sizeof(size_t)));
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
        SAFE_PIPE_CHECK(readWrapper(fd, m->content, m->size));
    }
#ifdef DEBUG
    printf("REC:done:%d\n", fd);
#endif
    return 0;
}

/**
 * @brief Will read count bytes into the buffer, or return an error.
 *
 * @param fd the socket from which to read.
 * @param buf the buffer in which to read.
 * @param count how many bytes to read.
 * @return int 0 on success, -1 on error.
 */
int readWrapper(int fd, void *buf, size_t count)
{
    ssize_t curLoc = 0;
    ssize_t tmp;

    // Since read isn't guaranteed to recieve all bytes, we need to call it in a loop until we read all required bytes.
    while (curLoc < count)
    {
        tmp = read(fd, buf + curLoc, count - curLoc);
        if (tmp <= 0)
        {
#ifdef DEBUG
            printf("Error when reading; err:%ld, Expected to read:%ld, Actually read:%ld\n", tmp, count, curLoc);
#endif
            return -1;
        }
        curLoc += tmp;
    }

    return 0;
}

/**
 * @brief Will write count bytes into the buffer, or return an error.
 *
 * @param fd the socket in which to write.
 * @param buf the buffer from which to read.
 * @param count how many bytes to write.
 * @return int 0 on success, -1 on error.
 */
int writeWrapper(int fd, void *buf, size_t count)
{
    ssize_t curLoc = 0;
    ssize_t tmp;

    // Since write isn't guaranteed to write all bytes, we need to call it in a loop until we write all required bytes.
    while (curLoc < count)
    {
        tmp = write(fd, buf + curLoc, count - curLoc);
        if (tmp <= 0)
        {
#ifdef DEBUG
            printf("Error when writing; err:%ld, Expected to write:%ld, Actually wrote:%ld\n", tmp, count, curLoc);
#endif
            return -1;
        }
        curLoc += tmp;
    }

    return 0;
}