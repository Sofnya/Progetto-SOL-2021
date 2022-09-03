#include "SERVER/globals.h"
#include "COMMON/macros.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

char SOCK_NAME[UNIX_PATH_MAX] = "default_address";
char LOG_FILE[UNIX_PATH_MAX] = "log";
int64_t CORE_POOL_SIZE = 8;
int64_t MAX_POOL_SIZE = UINT64_MAX;
int64_t MAX_FILES = 100;
int64_t MAX_MEMORY = 100 * 1024 * 1024;
int ENABLE_COMPRESSION = 1;
int VERBOSE_PRINT = 0;
int POLICY = P_LRU;

void load_config(char *path)
{
    FILE *configFile;
    char *line = NULL, *option = NULL, *value = NULL, *saveptr = NULL, *endptr = NULL;
    size_t len = 0;
    int lineN = 0;
    long long tmp;

    UNSAFE_NULL_CHECK(configFile = fopen(path, "r"));

    while (getline(&line, &len, configFile) != -1)
    {
        lineN++;
        len = strlen(line);

        // Stripping of the newlines.
        if (len >= 2)
        {
            if (line[len - 1] == '\n')
                line[len - 1] = '\00';
        }

        // Ignoring comments
        if (!strncmp(line, "//", 2))
        {
            continue;
        }

        option = strtok_r(line, "=", &saveptr);
        value = strtok_r(NULL, "=", &saveptr);

        if ((option == NULL) | (value == NULL))
        {
            printf("Malformed config file on line %d, continuing anyway.\n", lineN);
        }
        else if (!strcmp(option, "SOCK_NAME"))
        {
            strncpy(SOCK_NAME, value, UNIX_PATH_MAX - 1);
            SOCK_NAME[UNIX_PATH_MAX - 1] = '\00';
        }
        else if (!strcmp(option, "LOG_FILE"))
        {
            strncpy(LOG_FILE, value, UNIX_PATH_MAX - 1);
            LOG_FILE[UNIX_PATH_MAX - 1] = '\00';
        }
        else if (!strcmp(option, "CORE_POOL_SIZE"))
        {
            errno = 0;
            tmp = strtoll(value, NULL, 0);
            if (errno != 0 || tmp < 0)
            {
                perror("Error");
                printf("Invalid value on line %d, ignoring.\n", lineN);
                continue;
            }
            CORE_POOL_SIZE = tmp;
        }
        else if (!strcmp(option, "MAX_POOL_SIZE"))
        {
            errno = 0;
            tmp = strtoll(value, NULL, 0);
            if (errno != 0)
            {
                perror("Error");
                printf("Invalid value on line %d, ignoring.\n", lineN);
                continue;
            }
            if (tmp >= 0)
                MAX_POOL_SIZE = tmp;
            else
                MAX_POOL_SIZE = UINT64_MAX;
        }
        else if (!strcmp(option, "MAX_FILES"))
        {
            tmp = strtoll(value, &endptr, 0);
            if (tmp == __LONG_LONG_MAX__ || tmp <= 0)
            {
                printf("Invalid value on line %d, ignoring.\n", lineN);
            }
            else
            {
                MAX_FILES = tmp;

                if (!strncasecmp(endptr, "k", 1))
                    MAX_FILES *= 1024;
                else if (!strncasecmp(endptr, "m", 1))
                    MAX_FILES *= (1024 * 1024);
                else if (!strncasecmp(endptr, "g", 1))
                    MAX_FILES *= (1024 * 1024 * 1024);

                endptr = NULL;
            }
        }
        else if (!strcmp(option, "MAX_MEMORY"))
        {
            tmp = strtoll(value, &endptr, 0);
            if (tmp == __LONG_LONG_MAX__ || tmp <= 0)
            {
                printf("Invalid value on line %d, ignoring.\n", lineN);
            }
            else
            {
                MAX_MEMORY = tmp;

                if (!strncasecmp(endptr, "k", 1))
                    MAX_MEMORY *= 1024;
                else if (!strncasecmp(endptr, "m", 1))
                    MAX_MEMORY *= (1024 * 1024);
                else if (!strncasecmp(endptr, "g", 1))
                    MAX_MEMORY *= (1024 * 1024 * 1024);

                endptr = NULL;
            }
        }
        else if (!strcmp(option, "ENABLE_COMPRESSION"))
        {
            if (!strncmp(value, "TRUE", 4))
            {
                ENABLE_COMPRESSION = 1;
            }
            else if (!strncmp(value, "FALSE", 5))
            {
                ENABLE_COMPRESSION = 0;
            }
            else
            {
                printf("Invalid value %s on line %d, ignoring.\n", value, lineN);
            }
        }
        else if (!strcmp(option, "VERBOSE_PRINT"))
        {
            if (!strncmp(value, "TRUE", 4))
            {
                VERBOSE_PRINT = 1;
            }
            else if (!strncmp(value, "FALSE", 5))
            {
                VERBOSE_PRINT = 0;
            }
            else
            {
                printf("Invalid value %s on line %d, ignoring.\n", value, lineN);
            }
        }
        else if (!strcmp(option, "POLICY"))
        {
            if (!strncmp(value, "RANDOM*", 7))
            {
                POLICY = P_RAND;
            }
            else if (!strncmp(value, "FIFO", 4))
            {
                POLICY = P_FIFO;
            }
            else if (!strncmp(value, "LIFO", 4))
            {
                POLICY = P_LIFO;
            }
            else if (!strncmp(value, "LRU", 3))
            {
                POLICY = P_LRU;
            }
            else if (!strncmp(value, "MRU", 3))
            {
                POLICY = P_MRU;
            }
            else if (!strncmp(value, "LU", 2))
            {
                POLICY = P_LU;
            }
            else if (!strncmp(value, "MU", 2))
            {
                POLICY = P_MU;
            }
            else if (!strncmp(value, "SMOL", 4))
            {
                POLICY = P_SMOL;
            }
            else if (!strncmp(value, "BIGG", 4))
            {
                POLICY = P_BIGG;
            }
            else
            {
                printf("Invalid policy %s on line %d, ignoring.\n", value, lineN);
            }
        }
        else
        {
            printf("Unsupported option %s at line %d, ignoring.\n", option, lineN);
        }

        free(line);
        line = NULL;
        len = 0;
        saveptr = NULL;
    }
    free(line);
    fclose(configFile);
    puts("Succesfully loaded config:");
    printf("SOCK_NAME:%s\nCORE_POOL_SIZE:%ld\nMAX_POOL_SIZE:%ld\nMAX_FILES:%ld\nMAX_MEMORY:%ld\nENABLE_COMPRESSION:%d\nVERBOSE_PRINT:%d\nLOG_FILE:%s\n", SOCK_NAME, CORE_POOL_SIZE, MAX_POOL_SIZE, MAX_FILES, MAX_MEMORY, ENABLE_COMPRESSION, VERBOSE_PRINT, LOG_FILE);

    switch (POLICY)
    {
    case (P_RAND):
        puts("POLICY:RANDOM");
        break;

    case (P_FIFO):
        puts("POLICY:FIFO");
        break;

    case (P_LIFO):
        puts("POLICY:LIFO");
        break;

    case (P_LRU):
        puts("POLICY:LRU");
        break;

    case (P_MRU):
        puts("POLICY:MRU");
        break;

    case (P_MU):
        puts("POLICY:MU");
        break;

    case (P_LU):
        puts("POLICY:LU");
        break;

    case (P_SMOL):
        puts("POLICY:SMOL");
        break;

    case (P_BIGG):
        puts("POLICY:BIGG");
        break;

    default:
        puts("POLICY:INVALID");
        break;
    }
}