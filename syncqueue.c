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
    struct _list *cur = queue->_head, *next = NULL;
    
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


void syncqueuePush(void *el, SyncQueue *queue)
{
    PTHREAD_CHECK(pthread_mutex_lock(&queue->_mtx));
    if(!queue->_isOpen)
    {
        PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
        return;
    }
    if(queue->_head == NULL)
    {
        NULL_CHECK(queue->_head = malloc(sizeof(struct _list)));
        queue->_tail = queue->_head;
        queue->_tail->next = NULL;
        queue->_tail->prev = NULL;
        queue->_tail->data = el;
        queue->_len ++;
    }
    else 
    {
        NULL_CHECK(queue->_tail->next = malloc(sizeof(struct _list)));
        queue->_tail->next->prev = queue->_tail;
        queue->_tail = queue->_tail->next;
        queue->_tail->next = NULL;
        queue->_tail->data = el;
        queue->_len ++;
    }

    PTHREAD_CHECK(pthread_cond_signal(&queue->_newEl));
    PTHREAD_CHECK(pthread_mutex_unlock(&queue->_mtx));
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


int syncqueueLen(SyncQueue queue)
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
