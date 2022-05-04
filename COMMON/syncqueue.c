#include "syncqueue.h"
#include "macros.h"


void syncqueueInit(SyncQueue *queue)
{
    queue->_head = NULL;
    queue->_tail = NULL;
    queue->_len = 0;
    queue->_isOpen = true;

    PTHREAD_CHECK(pthread_mutex_init(&queue->_mtx, NULL));
    PTHREAD_CHECK(pthread_cond_init(&queue->_newEl, NULL));
}

void syncqueueDestroy(SyncQueue *queue)
{
    struct _slist *cur = queue->_head, *next = NULL;
    
    while(cur != NULL)
    {
        next = cur->next;
        free(cur);
        cur = next;
    }

    queue->_head = NULL;
    queue->_tail = NULL;
    queue->_len = 0;

    PTHREAD_CHECK(pthread_mutex_destroy(&queue->_mtx));
    PTHREAD_CHECK(pthread_cond_destroy(&queue->_newEl));
    
}


int syncqueuePush(void *el, SyncQueue *queue)
{
    PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));
    if(!queue->_isOpen)
    {
        PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
        errno = EINVAL;
        return -1;
    }
    if(queue->_head == NULL)
    {
        SAFE_NULL_CHECK(queue->_head = malloc(sizeof(struct _slist)));
        queue->_tail = queue->_head;
        queue->_head->next = NULL;
        queue->_head->prev = NULL;
        queue->_head->data = el;
        queue->_len ++;
    }
    else 
    {
        SAFE_NULL_CHECK(queue->_head->prev = malloc(sizeof(struct _slist)));
        queue->_head->prev->next = queue->_head;
        queue->_head = queue->_head->prev;
        queue->_head->prev = NULL;
        queue->_head->data = el;
        queue->_len ++;
    }

    PTHREAD_CHECK(pthread_cond_signal(&queue->_newEl));
    PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));

    return 0;
}

void *syncqueuePop(SyncQueue *queue)
{
    PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));
    while(queue->_len == 0) { 
        if(!queue->_isOpen){
            PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
            errno = EINVAL;
            return NULL;
        }
        PTHREAD_CHECK(pthread_cond_wait(&queue->_newEl, &queue->_mtx)); 
    }

    void *el = queue->_tail->data;
    
    if(queue->_tail->prev == NULL)
    {
        free(queue->_tail);
        queue->_tail = NULL;
        queue->_head = NULL;
        queue->_len --;
    }
    else{
        queue->_tail = queue->_tail->prev;
        free(queue->_tail->next);
        queue->_tail->next = NULL;
        queue->_len --;
    }
    PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
    
    return el;
}


long syncqueueLen(SyncQueue queue)
{
    return queue._len;
}


void syncqueueClose(SyncQueue *queue)
{
    PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));
    queue->_isOpen = false;
    PTHREAD_CHECK(pthread_cond_broadcast(&queue->_newEl));
    PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
}
