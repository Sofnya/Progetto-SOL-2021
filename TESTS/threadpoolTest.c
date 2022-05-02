#include <stdio.h>
#include <stdlib.h>

#include "threadpool.h"


void test(void *par)
{
    int p = *(int *)par;
    free(par);
    printf("HI!:%d\n", p);
    return;
}


int main(int argc, char const *argv[])
{
    ThreadPool pool;
    int i;
    threadpoolInit(4, &pool);

    for(i = 0; i < 100; i++)
    {
        int *j = malloc(sizeof(int));
        *j = i;
        threadpoolSubmit(&test, j, &pool);
    }

    threadpoolCleanExit(&pool);
    return 0;
}

