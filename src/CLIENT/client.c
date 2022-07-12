#include <stdlib.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <dirent.h>

#include "CLIENT/api.h"
#include "COMMON/hashtable.h"
#include "COMMON/macros.h"

#define UNIX_PATH_MAX 108
#define SOCKNAME "fakeAddress"
#define N 100

int writeNFiles(char *dir, int n, char *missDirName, int delay);

int main(int argc, char *argv[])
{
    int opt;
    int n;
    long delay = 0;
    char *missDirName = NULL, *readDirName = NULL, *sockname = NULL, *tmp;
    HashTable openFiles;
    void *flag;
    FILE *file;

    hashTableInit(50, &openFiles);

    while ((opt = getopt(argc, argv, "hf:w:W:D:r:R::d:t:l:u:c:p")) != -1)
    {
        switch (opt)
        {
        case ('h'):
        {
            puts("Usage:");
            puts("client -h show this help");
            puts("client -f [sockname] set the sock name to connect to");
            puts("client -w [dirname,n=-1] write up to n files from given dir to server");
            puts("client -W [filename, ... ,filename] write all given files to server");
            puts("client -D [dir] set missDir");
            puts("client -r [fileName] read fileName from server");
            puts("client -R [N] read N files from server");
            puts("client -d [dir] set readDir");
            puts("client -t [n] set delay between requests");
            puts("client -l [filename, ... , filename] locks given files");
            puts("client -u [filename, ... , filename] unlocks given files");
            puts("client -c [filename, ... , filename] removes given files");
            puts("client -p enables verbose printing");
            exit(EXIT_SUCCESS);
        }
        case ('f'):
        {
            struct timespec abstime;

            if (sockname != NULL)
            {
                usleep(delay);
                closeConnection(sockname);
            }
            sockname = optarg;

            timespec_get(&abstime, TIME_UTC);
            abstime.tv_sec += 10;

            openConnection(sockname, 100, abstime);
            break;
        }
        case ('w'):
        {
            n = 0;
            if (strchr(optarg, ',') != NULL)
            {
                tmp = strtok(optarg, ",");
                n = atoi(strtok(NULL, ","));

                if (strtok(NULL, ",") != NULL)
                {
                    puts("Too many options after -w");
                    exit(EXIT_FAILURE);
                }
            }
            else
            {
                tmp = optarg;
            }
            if (n == 0)
                n = INT32_MAX;

            printf("Wrote %d files\n", writeNFiles(tmp, n, missDirName, delay));
            break;
        }
        case ('W'):
        {
            tmp = strtok(optarg, ",");
            do
            {
                usleep(delay);
                if ((create_file(tmp, O_CREATE | O_LOCK, missDirName) == -1) && verbose)
                {
                    printf("Couldn't create file %s\n", tmp);
                    continue;
                }
                usleep(delay);
                if (writeFile(tmp, missDirName) == -1 && verbose)
                {
                    printf("Couldn't write file %s\n", tmp);
                }

                usleep(delay);
                closeFile(tmp);

            } while ((tmp = strtok(NULL, ",")) != NULL);
            break;
        }
        case ('D'):
        {
            missDirName = optarg;

            if (verbose)
            {
                printf("MissDir set to %s\n", missDirName);
            }
            break;
        }
        case ('r'):
        {
            void *buf;
            size_t size;
            tmp = strtok(optarg, ",");
            do
            {
                if (hashTableGet(tmp, &flag, openFiles) == -1)
                {
                    usleep(delay);
                    if (openFile(tmp, 0) == -1 && verbose)
                    {
                        printf("Couldn't open file %s\n", tmp);
                    }
                }
                usleep(delay);
                if (readFile(tmp, &buf, &size) == -1 && verbose)
                {
                    printf("Couldn't read file %s\n", tmp);
                }
                else
                {
                    file = fopen(tmp, "w+");
                    fwrite(buf, size, 1, file);
                    fclose(file);
                    free(buf);
                }
                if (hashTableGet(tmp, &flag, openFiles) == -1)
                {
                    usleep(delay);
                    if (closeFile(tmp) == -1 && verbose)
                    {
                        printf("Couldn't close file %s\n", tmp);
                    }
                }

            } while ((tmp = strtok(NULL, ",")) != NULL);
            break;
        }

        case ('R'):
        {
            n = 0;
            if (optarg != NULL)
            {
                n = atoi(optarg);
            }
            if (verbose)
            {
                printf("Reading %d files\n", n);
            }
            usleep(delay);
            readNFiles(n, readDirName);
            break;
        }
        case ('d'):
        {
            readDirName = optarg;
            if (verbose)
            {
                printf("ReadDir set to %s\n", readDirName);
            }
            break;
        }
        case ('t'):
        {
            delay = atol(optarg);
            if (verbose)
            {
                printf("delay set:%ldms\n", delay);
            }
            delay *= 1000;
            break;
        }
        case ('l'):
        {

            tmp = strtok(optarg, ",");
            do
            {
                usleep(delay);
                if (openFile(tmp, 0) == -1 && verbose)
                {
                    printf("Couldn't open file %s\n", tmp);
                }
                usleep(delay);
                if (lockFile(tmp) == -1 && verbose)
                {
                    printf("Couldn't lock file %s\n", tmp);
                }
                else
                {
                    hashTablePut(tmp, NULL, openFiles);
                }
            } while ((tmp = strtok(NULL, ",")) != NULL);
            break;
        }
        case ('u'):
        {
            tmp = strtok(optarg, ",");
            do
            {
                usleep(delay);

                if (unlockFile(tmp) == -1 && verbose)
                {
                    printf("Couldn't unlock file %s\n", tmp);
                }
                else
                {
                    usleep(delay);
                    closeFile(tmp);

                    hashTableRemove(tmp, NULL, openFiles);
                }
            } while ((tmp = strtok(NULL, ",")) != NULL);
            break;
        }
        case ('c'):
        {
            tmp = strtok(optarg, ",");
            do
            {
                if (hashTableGet(tmp, &flag, openFiles) == -1)
                {
                    usleep(delay);
                    openFile(tmp, O_LOCK);
                    if (verbose)
                    {
                        printf("Opened file %s for removal.\n", tmp);
                    }
                }
                usleep(delay);
                if (removeFile(tmp) == -1 && verbose)
                {
                    printf("Couldn't remove file %s\n", tmp);
                }
                else if (verbose)
                {
                    printf("Removed file %s\n", tmp);
                }

            } while ((tmp = strtok(NULL, ",")) != NULL);
            puts(optarg);
            break;
        }
        case ('p'):
        {
            puts("Verbose print activated");
            verbose = 1;
            break;
        }
        default:
        {
            puts("Unrecognized!");
            break;
        }
        }
    }

    hashTableDestroy(&openFiles);
}

/**
 * @brief Visits the directory dirPath and all its subdirectories untill its written N files from within them.
 *
 * @param dirPath the directory to visit recuriìsively.
 * @param n the number of files to write.
 * @param missDirName where to write files recieved back from the server.
 * @return int the number of files actually written.
 */
int writeNFiles(char *dirPath, int n, char *missDirName, int delay)
{
    int count = 0;
    DIR *dir;
    struct dirent *cur;
    char *path;

    if (verbose)
    {
        printf("Visiting directory: %s with n=%d\n", dirPath, n);
    }

    dir = opendir(dirPath);
    if (dir == NULL)
        return 0;

    while (count < n)
    {
        cur = readdir(dir);
        if (cur == NULL)
            break;

        if (cur->d_type == DT_DIR)
        {
            if (strcmp(cur->d_name, ".") != 0 && strcmp(cur->d_name, "..") != 0)
            {
                path = malloc(strlen(dirPath) + strlen(cur->d_name) + 100);
                sprintf(path, "%s/%s", dirPath, cur->d_name);
                count += writeNFiles(path, n - count, missDirName, delay);
                free(path);
            }
        }
        else if (cur->d_type == DT_REG)
        {
            path = malloc(strlen(dirPath) + strlen(cur->d_name) + 100);
            sprintf(path, "%s/%s", dirPath, cur->d_name);

            usleep(delay);
            if ((create_file(path, O_CREATE | O_LOCK, missDirName) == -1))
            {
                if (verbose)
                {
                    printf("Couldn't create file %s\n", path);
                }
                free(path);
                continue;
            }
            usleep(delay);
            if (writeFile(path, missDirName) == -1 && verbose)
            {
                printf("Couldn't write file %s\n", path);
            }
            else
                count++;

            usleep(delay);
            closeFile(path);
            free(path);
        }
    }
    closedir(dir);
    return count;
}
