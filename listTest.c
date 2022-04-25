#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "list.h"
#include "macros.h"

int main(int argc, char const *argv[])
{
    List list, list2;
    int i;
    int *j;
    
    listInit(&list);
    listInit(&list2);

    for(i=0; i <= 9999; i++)
    {
        UNSAFE_NULL_CHECK(j = malloc(sizeof(int)));
        *j = i;
        //printf("Inserting %d...\n", *j);
        ERROR_CHECK(listAppend((void *)j, &list));
    }
    puts("Appends done!");

    UNSAFE_NULL_CHECK(j = malloc(sizeof(int)));
    *j = 1337;
    listPut(0, (void *)j, &list);
    puts("Put done");

    printf("aaa: %d\n", *j);
    puts("lol i wish");
    for(i = 0; i <= 9999; i++)
    {
        ERROR_CHECK(listGet(i + 1, (void **)&j, &list));
        printf("%d: %d ok!\n", i, *j);
        assert(i == *j);
        free(j);
    }

    puts("Gets done");

    listPop((void **)&j, &list);
    assert(*j == 1337);

    puts("Pop done");

    free(j);

    listDestroy(&list);
    puts("Destroy done");


    for(i = 0; i <= 9999; i++)
    {
        UNSAFE_NULL_CHECK(j = malloc(sizeof(int)));
        *j = i;
        ERROR_CHECK(listPush(j, &list2));
    }

    for(i = 9999; i >= 0; i--)
    {
        ERROR_CHECK(listPop(&j, &list2));
        assert(*j == i);
        free(j);
    }

    listDestroy(&list2);
    return 0;
}
