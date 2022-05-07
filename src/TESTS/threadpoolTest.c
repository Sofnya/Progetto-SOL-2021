#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#include "COMMON/threadpool.h"


void test(void *par)
{
    int p = *(int *)par;
    free(par);
    printf("HI!:%d\n", p);
    usleep(10000);
    return;
}


int main(int argc, char const *argv[])
{
    ThreadPool pool;
    int i;
    threadpoolInit(1, 300, &pool);

    for(i = 0; i < 1000; i++)
    {
        int *j = malloc(sizeof(int));
        *j = i;
        threadpoolSubmit(&test, j, &pool);
        usleep(1000);
    }

    threadpoolCleanExit(&pool);
    threadpoolDestroy(&pool);
    return 0;
}

