#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "CLIENT/clientHelpers.h"
#include "COMMON/fileContainer.h"
#include "CLIENT/api.h"

void __mkdir(char *path)
{
    char *prev, *cur;
    prev = strtok(path, "/");
    while ((cur = strtok(NULL, "/")) != NULL)
    {
        if (verbose)
        {
            printf("Making dir:%s\n", prev);
        }

        if (mkdir(prev, 0777) && errno != EEXIST)
            perror("error while creating dir");
        prev[strlen(prev)] = '/';
    }
}
void __writeToDir(FileContainer *fc, uint64_t size, const char *dirname)
{
    int i;
    char *path;
    FILE *file;

    if (dirname == NULL)
        return;

    if (verbose)
    {
        printf("Writing %ld files to dir:%s\n", size, dirname);
    }

    for (i = 0; i < size; i++)
    {
        path = malloc(strlen(fc[i].name) + 10 + strlen(dirname));
        sprintf(path, "%s/%s", dirname, fc[i].name);

        __mkdir(path);

        if ((file = fopen(path, "w+")) != NULL)
        {
            fwrite(fc[i].content, 1, fc[i].size, file);
            fclose(file);

            if (verbose)
            {
                printf("Wrote %s\n", path);
            }
        }
        else
        {
            perror("Couldn't open local path");
        }

        free(path);
        destroyContainer(&fc[i]);
    }
}

void __writeBufToDir(void *buf, uint64_t size, const char *fileName, const char *dirname)
{
    char *path;
    FILE *file;

    if (dirname == NULL)
        return;

    if (verbose)
    {
        printf("Writing file %s to dir:%s\n", fileName, dirname);
    }

    path = malloc(strlen(fileName) + 10 + strlen(dirname));
    sprintf(path, "%s/%s", dirname, fileName);

    __mkdir(path);

    if ((file = fopen(path, "w+")) != NULL)
    {
        fwrite(buf, 1, size, file);
        fclose(file);

        if (verbose)
        {
            printf("Wrote %s\n", path);
        }
    }
    else
    {
        perror("Couldn't open local path");
    }

    free(path);
}