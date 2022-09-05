#include "COMMON/syncqueue.h"
#include "COMMON/macros.h"

/**
 * @brief Initializes given SyncQueue.
 *
 * @param queue the SyncQueue to initialize.
 */
void syncqueueInit(SyncQueue *queue)
{
    queue->_head = NULL;
    queue->_tail = NULL;
    queue->_len = 0;
    queue->_isOpen = true;

    VOID_PTHREAD_CHECK(pthread_mutex_init(&queue->_mtx, NULL));
    VOID_PTHREAD_CHECK(pthread_cond_init(&queue->_newEl, NULL));
}

/**
 * @brief Destroyes given SyncQueue, freeing it's resources.
 *
 * @param queue the SyncQueue to destroy.
 */
void syncqueueDestroy(SyncQueue *queue)
{
    struct _slist *cur = queue->_head, *next = NULL;

    while (cur != NULL)
    {
        next = cur->next;
        free(cur);
        cur = next;
    }

    queue->_head = NULL;
    queue->_tail = NULL;
    queue->_len = 0;

    VOID_PTHREAD_CHECK(pthread_mutex_destroy(&queue->_mtx));
    VOID_PTHREAD_CHECK(pthread_cond_destroy(&queue->_newEl));
}

/**
 * @brief Pushes given element on given SyncQueue.
 *
 * @param el the element to push.
 * @param queue the SyncQueue on which to push given element.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int syncqueuePush(void *el, SyncQueue *queue)
{
    PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));

    // If the SyncQueue is closed we return an error.
    if (!queue->_isOpen)
    {
        PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
        errno = EINVAL;
        return -1;
    }
    if (queue->_head == NULL)
    {
        SAFE_NULL_CHECK(queue->_head = malloc(sizeof(struct _slist)));
        queue->_tail = queue->_head;
        queue->_head->next = NULL;
        queue->_head->prev = NULL;
        queue->_head->data = el;
        queue->_len++;
    }
    else
    {
        SAFE_NULL_CHECK(queue->_head->prev = malloc(sizeof(struct _slist)));
        queue->_head->prev->next = queue->_head;
        queue->_head = queue->_head->prev;
        queue->_head->prev = NULL;
        queue->_head->data = el;
        queue->_len++;
    }

    // Once pushed we wake a thread waiting for a new element, if present.
    PTHREAD_CHECK(pthread_cond_signal(&queue->_newEl));
    PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));

    return 0;
}

/**
 * @brief Pops an element from given SyncQueue, blocking until one is available.
 *
 * @param queue the SyncQueue to query.
 * @return void* the element if successfull, NULL and sets errno otherwise.
 */
void *syncqueuePop(SyncQueue *queue)
{
    NULL_PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));

    // Wait until at least one element is available.
    while (queue->_len == 0)
    {
        // Can't wait on a closed, empty, SyncQueue.
        if (!queue->_isOpen)
        {
            NULL_PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
            errno = EINVAL;
            return NULL;
        }

        // If the SyncQueue is still open, but empty, we wait for new elements to be pushed.
        NULL_PTHREAD_CHECK(pthread_cond_wait(&queue->_newEl, &queue->_mtx));
    }

    // Now just pop the last element of the SyncQueue.
    void *el = queue->_tail->data;

    if (queue->_tail->prev == NULL)
    {
        free(queue->_tail);
        queue->_tail = NULL;
        queue->_head = NULL;
        queue->_len--;
    }
    else
    {
        queue->_tail = queue->_tail->prev;
        free(queue->_tail->next);
        queue->_tail->next = NULL;
        queue->_len--;
    }
    NULL_PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));

    return el;
}

/**
 * @brief Returns the length of given SyncQueue.
 *
 * @param queue the SyncQueue to query.
 * @return long the length of given SyncQueue.
 */
long syncqueueLen(SyncQueue queue)
{
    return queue._len;
}

/**
 * @brief Closes given SyncQueue, stopping it from accepting new elements and waking all threads waiting on a syncqueuePop.
 *
 * @param queue the SyncQueue to close.
 */
void syncqueueClose(SyncQueue *queue)
{
    VOID_PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));
    queue->_isOpen = false;
    VOID_PTHREAD_CHECK(pthread_cond_broadcast(&queue->_newEl));
    VOID_PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
}

/**
 * @brief Opens given SyncQueue, making it accept new elements again, and waking all threads waiting on a syncqueuePop.
 *
 * @param queue the SyncQueue to open.
 */
void syncqueueOpen(SyncQueue *queue)
{
    VOID_PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));
    queue->_isOpen = true;
    VOID_PTHREAD_CHECK(pthread_cond_broadcast(&queue->_newEl));
    VOID_PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
}
