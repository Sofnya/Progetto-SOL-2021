#include <stdio.h>
#include <assert.h>

#include "syncqueue.h"




int main(int argc, char const *argv[])
{
    syncQueue *queue;
    int i, j = 99999;

    NULL_CHECK(queue = malloc(sizeof(syncQueue)));
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
    return 0;
}

