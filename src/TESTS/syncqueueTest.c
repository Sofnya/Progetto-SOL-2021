#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "COMMON/syncqueue.h"
#include "COMMON/macros.h"




int main(int argc, char const *argv[])
{
    SyncQueue *queue;
    int i, j = 99999;

    UNSAFE_NULL_CHECK(queue = malloc(sizeof(SyncQueue)));
    syncqueueInit(queue);
    for(i = 0; i < 100000; i++)
    {
        syncqueuePush(&i, queue);
        assert(*(int*)syncqueuePop(queue) == i);
    }
    for(i = 0; i < 999; i++)
    {
        syncqueuePush(&j, queue);
    }
    for(i = 0; i < 900; i++)
    {
        assert(j == *(int*)syncqueuePop(queue));
    }
    puts("Tests succesfull!");

    syncqueueDestroy(queue);
    free(queue);
    return 0;
}

