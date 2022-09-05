#include <stdlib.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <dirent.h>
#include <signal.h>

#include "CLIENT/api.h"
#include "CLIENT/clientHelpers.h"
#include "COMMON/hashtable.h"
#include "COMMON/macros.h"

#define UNIX_PATH_MAX 108
#define SOCKNAME "defaultAddress"
#define N 100

int writeNFiles(char *dir, int n, char *missDirName, int delay);

char *lockedFile = NULL;

int main(int argc, char *argv[])
{
    int opt;
    int n;
    long delay = 0;
    char *missDirName = NULL, *readDirName = NULL, *sockname = NULL, *tmp;
    void *flag;

    // Ignore SIGPIPE to avoid problems with read/writes on pipes.
    sigaction(SIGPIPE, &(struct sigaction){{SIG_IGN}}, NULL);

    // Now start parsing all commands.
    while ((opt = getopt(argc, argv, "hf:w:W:D:r:R::d:t:l:u:c:p")) != -1)
    {
        switch (opt)
        {
            // Help.
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
        // Connect.
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
        // WriteN.
        case ('w'):
        {
            n = 0;
            // Parse the argument to find dirname and the number of elements to write.
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
        // Write.
        case ('W'):
        {
            char *saveptr;
            tmp = strtok_r(optarg, ",", &saveptr);
            do
            {
                usleep(delay);

                // Try to create the file first.
                if ((create_file(tmp, O_CREATE | O_LOCK, missDirName) == -1) && verbose)
                {
                    printf("Couldn't create file %s\n", tmp);
                    continue;
                }
                usleep(delay);

                // If file was successfully created, write it's contents.
                if (writeFile(tmp, missDirName) == -1 && verbose)
                {
                    printf("Couldn't write file %s\n", tmp);
                }

                // Finally close the file.
                usleep(delay);
                closeFile(tmp);

            } while ((tmp = strtok_r(NULL, ",", &saveptr)) != NULL);
            break;
        }
        // Set MissDir.
        case ('D'):
        {
            missDirName = optarg;

            if (verbose)
            {
                printf("MissDir set to %s\n", missDirName);
            }
            break;
        }
        // Read.
        case ('r'):
        {
            void *buf;
            char *saveptr;
            size_t size;

            // Need to read all files in the args, which are comma-separated.
            tmp = strtok_r(optarg, ",", &saveptr);
            do
            {
                // Open the file if it's not already our locked file.
                if (lockedFile != NULL && !(strcmp(tmp, lockedFile)))
                {
                    usleep(delay);
                    if (openFile(tmp, 0) == -1)
                    {
                        if (verbose)
                        {
                            printf("Couldn't open file %s\n", tmp);
                        }
                        continue;
                    }
                }

                // The actual read.
                usleep(delay);
                if (readFile(tmp, &buf, &size) == -1)
                {
                    if (verbose)
                    {
                        printf("Couldn't read file %s\n", tmp);
                    }
                }
                else
                {
                    // If we managed to read the file, write it to disk inside readDir.
                    __writeBufToDir(buf, size, tmp, readDirName);
                    free(buf);
                }

                // If the file wasn't locked before, then we opened it for the read, and should now close it again.
                if (lockedFile == NULL || strcmp(tmp, lockedFile))
                {
                    usleep(delay);
                    if (closeFile(tmp) == -1 && verbose)
                    {
                        if (verbose)
                        {
                            printf("Couldn't close file %s\n", tmp);
                        }
                    }
                }

            } while ((tmp = strtok_r(NULL, ",", &saveptr)) != NULL);
            break;
        }
        // ReadN.
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
        // Set readDir.
        case ('d'):
        {
            readDirName = optarg;
            if (verbose)
            {
                printf("ReadDir set to %s\n", readDirName);
            }
            break;
        }
        // Set delay.
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
        // Lock file.
        case ('l'):
        {
            char *saveptr;

            // Need to lock all comma-separated files.
            tmp = strtok_r(optarg, ",", &saveptr);
            do
            {
                usleep(delay);
                if (openFile(tmp, 0) == -1)
                {
                    if (verbose)
                    {
                        printf("Couldn't open file %s\n", tmp);
                    }
                    continue;
                }
                usleep(delay);
                if (lockFile(tmp) == -1)
                {
                    if (verbose)
                    {
                        printf("Couldn't lock file %s\n", tmp);
                    }
                }
                else
                {
                    // Keep track of our locked file.
                    if (lockedFile != NULL)
                    {
                        free(lockedFile);
                        lockedFile = NULL;
                    }
                    UNSAFE_NULL_CHECK(lockedFile = malloc((strlen(tmp) + 1) * sizeof(char)));
                    strcpy(lockedFile, tmp);
                }
            } while ((tmp = strtok_r(NULL, ",", &saveptr)) != NULL);
            break;
        }
        // Unlock.
        case ('u'):
        {
            char *saveptr;

            // Unlock all comma-separated files.
            tmp = strtok_r(optarg, ",", &saveptr);
            do
            {
                usleep(delay);

                if (unlockFile(tmp) == -1)
                {
                    if (verbose)
                    {
                        printf("Couldn't unlock file %s\n", tmp);
                    }
                }
                else
                {
                    usleep(delay);
                    closeFile(tmp);

                    // Once unlocked and closed, remove it from locked file.
                    if (lockedFile != NULL)
                    {
                        free(lockedFile);
                        lockedFile = NULL;
                    }
                }
            } while ((tmp = strtok_r(NULL, ",", &saveptr)) != NULL);
            break;
        }
        // Remove.
        case ('c'):
        {
            char *saveptr;

            // Remove all comma-separated-files.
            tmp = strtok_r(optarg, ",", &saveptr);
            do
            {
                // If the file isn't already locked, open and lock it.
                if (lockedFile == NULL || !strcmp(tmp, lockedFile))
                {
                    usleep(delay);
                    if (openFile(tmp, O_LOCK) == -1)
                    {
                        if (verbose)
                        {
                            printf("Couldn't open file %s for removal.\n", tmp);
                        }
                        continue;
                    }
                    else
                    {
                        if (lockedFile != NULL)
                        {
                            free(lockedFile);
                            lockedFile = NULL;
                        }
                        UNSAFE_NULL_CHECK(lockedFile = malloc((strlen(tmp) + 1) * sizeof(char)));
                        strcpy(lockedFile, tmp);
                    }
                    if (verbose)
                    {
                        printf("Opened file %s for removal.\n", tmp);
                    }
                }
                usleep(delay);
                if (removeFile(tmp) == -1)
                {
                    if (verbose)
                    {
                        printf("Couldn't remove file %s\n", tmp);
                    }
                    continue;
                }
                else if (verbose)
                {
                    printf("Removed file %s\n", tmp);
                }

                // If we managed to remove our locked file, update it.
                if (lockedFile != NULL)
                {
                    free(lockedFile);
                    lockedFile = NULL;
                }

            } while ((tmp = strtok_r(NULL, ",", &saveptr)) != NULL);
            puts(optarg);
            break;
        }
        // Enable verbose print.
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

    puts("Exiting");
    if (sockname != NULL)
    {
        usleep(delay);
        closeConnection(sockname);
    }
    if (lockedFile != NULL)
    {
        free(lockedFile);
    }
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
            if (create_file(path, O_CREATE | O_LOCK, missDirName) == -1)
            {
                if (verbose)
                {
                    printf("Couldn't create file %s\n", path);
                }
                free(path);
                continue;
            }
            else
            {
                if (lockedFile != NULL)
                {
                    free(lockedFile);
                    lockedFile = NULL;
                }
            }
            usleep(delay);
            if (writeFile(path, missDirName) == -1)
            {
                if (verbose)
                {
                    printf("Couldn't write file %s\n", path);
                }
            }
            else
            {
                count++;
            }

            usleep(delay);
            if (closeFile(path) == -1)
            {
                if (verbose)
                {
                    printf("Couldn't close file %s\n", path);
                }
            }
            free(path);
        }
    }
    closedir(dir);
    return count;
}
