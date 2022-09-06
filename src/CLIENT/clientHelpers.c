#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "CLIENT/clientHelpers.h"
#include "COMMON/fileContainer.h"
#include "CLIENT/api.h"

/**
 * @brief Creates all missing directories in the path, ignoring the final element.
 *
 * @param path the path to the directory to create.
 */
void __mkdir(char *path)
{
    char *prev, *cur;
    char *saveptr;

    // We do some strtok magic. In prev we keep the path to our current directory, in cur the latest directory.
    prev = strtok_r(path, "/", &saveptr);
    while ((cur = strtok_r(NULL, "/", &saveptr)) != NULL)
    {

        if (mkdir(prev, 0777) && errno != EEXIST)
        {
            perror("error while creating dir");
        }
        // Since strok substitutes \00 to the token, we need to restore prev.
        prev[strlen(prev)] = '/';
    }
}

/**
 * @brief Writes all FileContainers in fc as Files in given path, does nothing if dirname is NULL.
 *
 * @param fc the array of FileContainers to write to disk.
 * @param size the size of fc.
 * @param dirname the path in which to write given FileContainers.
 */
void __writeToDir(FileContainer *fc, size_t size, const char *dirname)
{
    int i;
    char *path;
    FILE *file;

    if (dirname == NULL)
        return;

    // For every FileContainer.
    for (i = 0; i < size; i++)
    {
        // We first create the path of the new file, by appending it's name to dirname.
        path = malloc(strlen(fc[i].name) + 10 + strlen(dirname));
        sprintf(path, "%s/%s", dirname, fc[i].name);

        // We create all directories in path.
        __mkdir(path);

        // Create and open the file for writing.
        if ((file = fopen(path, "w+")) != NULL)
        {
            // And actually write it's contents to disk.
            fwrite(fc[i].content, 1, fc[i].size, file);
            fclose(file);
        }
        else
        {
            perror("Couldn't open local path");
        }

        free(path);
        destroyContainer(&fc[i]);
    }
}

/**
 * @brief Writes a file of content buf, and name fileName in directory dirname.
 *
 * @param buf the buffer from which to read the file's contents.
 * @param size the size of buf.
 * @param fileName the name of the file to write.
 * @param dirname the path in which to write the file.
 */
void __writeBufToDir(void *buf, size_t size, const char *fileName, const char *dirname)
{
    char *path;
    FILE *file;

    if (dirname == NULL)
        return;

    path = malloc(strlen(fileName) + 10 + strlen(dirname));
    sprintf(path, "%s/%s", dirname, fileName);

    __mkdir(path);

    if ((file = fopen(path, "w+")) != NULL)
    {
        fwrite(buf, 1, size, file);
        fclose(file);
    }
    else
    {
        perror("Couldn't open local path");
    }

    free(path);
}