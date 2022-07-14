#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "PARSER/parser.h"
#include "COMMON/macros.h"

int main(int argc, char const *argv[])
{
    Stats stats;
    if (argc != 2)
    {
        puts("Usage:\n parser logFile");
        exit(EXIT_SUCCESS);
    }

    stats = parse(argv[1]);

    printStats(stats);
    return 0;
}

Stats parse(const char *path)
{
    Stats stats;
    FILE *logFile;
    char *line = NULL, *preamble = NULL, *value = NULL, *saveptr = NULL, *type = NULL;
    char *innersaveptr = NULL, *value1 = NULL, *value2 = NULL;
    size_t len = 0;
    int lineN = 0;
    long long tmp1, tmp2;
    long long curOpen = 0;

    UNSAFE_NULL_CHECK(logFile = fopen(path, "r"));

    stats.closeN = 0;
    stats.readN = 0;
    stats.writeN = 0;
    stats.lockN = 0;
    stats.unlockN = 0;
    stats.openN = 0;
    stats.closeN = 0;

    stats.readSize = 0;
    stats.writeSize = 0;

    stats.nErrors = 0;

    stats.maxSize = 0;
    stats.maxN = 0;

    stats.missN = 0;
    stats.missRemoved = 0;

    stats.connN = 0;
    stats.maxConn = 0;
    stats.spaceSaved = 0;

    stats.nRequests = 0;
    stats.nResponse = 0;

    while (getline(&line, &len, logFile) != -1)
    {
        lineN++;
        len = strlen(line);

        // Stripping of the newlines.
        if (len >= 2)
        {
            if (line[len - 1] == '\n')
                line[len - 1] = '\00';
        }

        preamble = strtok_r(line, "\t", &saveptr);
        value = strtok_r(NULL, "\t", &saveptr);

        if ((preamble == NULL) || (value == NULL))
        {
            printf("Malformed log file on line %d, continuing anyway.\n", lineN);
        }
        else if (!strncmp(value, "[STATUS", 7))
        {
            // Ignored
        }
        else if (!strncmp(value, "[CONN_OPEN", 10))
        {
            curOpen++;
            if (curOpen > stats.maxConn)
            {
                stats.maxConn = curOpen;
            }
            stats.connN++;
        }
        else if (!strncmp(value, "[CONN_CLOSE", 11))
        {
            curOpen--;
        }
        else if (!strncmp(value, "[REQUEST", 8))
        {
            value1 = strtok_r(value, ">", &innersaveptr);
            type = strtok_r(NULL, ">", &innersaveptr);
            value1 = strtok_r(NULL, ">", &innersaveptr);
            value1 = strtok_r(NULL, ">", &innersaveptr);

            if (!strncmp(type, "READ", 4))
            {
                stats.readN++;
            }
            else if (!strncmp(type, "WRITE", 5))
            { // Get size, if present
                value1 = strtok_r(value1, ":", &innersaveptr);
                value1 = strtok_r(NULL, ":", &innersaveptr);

                tmp1 = atoll(value1);

                stats.writeN++;
                stats.writeSize += tmp1;
            }
            else if (!strncmp(type, "LOCK", 4))
            {
                stats.lockN++;
            }
            else if (!strncmp(type, "UNLOCK", 6))
            {
                stats.unlockN++;
            }
            else if (!strncmp(type, "OPEN", 4))
            {
                stats.openN++;
            }
            else if (!strncmp(type, "CLOSE", 5))
            {
                stats.closeN++;
            }

            stats.nRequests++;
        }
        else if (!strncmp(value, "[RESPONSE", 9))
        {
            value1 = strtok_r(value, ">", &innersaveptr);
            type = strtok_r(NULL, ">", &innersaveptr);
            value1 = strtok_r(NULL, ">", &innersaveptr);

            // Get size
            value1 = strtok_r(value1, ":", &innersaveptr);
            value1 = strtok_r(NULL, ":", &innersaveptr);

            tmp1 = atoll(value1);

            if (!strncmp(type, "READ", 4))
            {
                stats.readSize += tmp1;
            }
            if (!strncmp(type, "ERROR", 5))
            {
                stats.nErrors++;
            }
            stats.nResponse++;
        }
        else if (!strncmp(value, "[SIZE", 5))
        {
            value1 = strtok_r(value, ">", &innersaveptr);
            value1 = strtok_r(NULL, ">", &innersaveptr);
            value2 = strtok_r(NULL, ">", &innersaveptr);

            // Get curN value
            value1 = strtok_r(value1, ":", &innersaveptr);
            value1 = strtok_r(NULL, ":", &innersaveptr);

            tmp1 = atoll(value1);
            if (tmp1 > stats.maxN)
            {
                stats.maxN = tmp1;
            }

            // Get curSize value
            value2 = strtok_r(value2, ":", &innersaveptr);
            value2 = strtok_r(NULL, ":", &innersaveptr);

            tmp2 = atoll(value2);
            if (tmp2 > stats.maxSize)
            {
                stats.maxSize = tmp2;
            }
        }
        else if (!strncmp(value, "[COMPRESSED", 11))
        {
            value1 = strtok_r(value, ">", &innersaveptr);
            value1 = strtok_r(NULL, ">", &innersaveptr);
            value2 = strtok_r(NULL, ">", &innersaveptr);

            // Get from value
            value1 = strtok_r(value1, ":", &innersaveptr);
            value1 = strtok_r(NULL, ":", &innersaveptr);

            tmp1 = atoll(value1);

            // Get to value
            value2 = strtok_r(value2, ":", &innersaveptr);
            value2 = strtok_r(NULL, ":", &innersaveptr);

            tmp2 = atoll(value2);

            stats.spaceSaved += (tmp1 - tmp2);
        }
        else if (!strncmp(value, "[CAPMISS", 8))
        {
            value1 = strtok_r(value, ">", &innersaveptr);
            value1 = strtok_r(NULL, ">", &innersaveptr);
            value2 = strtok_r(NULL, ">", &innersaveptr);

            // Get to free
            value1 = strtok_r(value1, ":", &innersaveptr);
            value1 = strtok_r(NULL, ":", &innersaveptr);

            tmp1 = atoll(value1);

            // Get removed
            value2 = strtok_r(value2, ":", &innersaveptr);
            value2 = strtok_r(NULL, ":", &innersaveptr);

            tmp2 = atoll(value2);

            stats.missRemoved += tmp2;
            stats.missN++;
        }
        else
        {
            printf("Unsupported option %s at line %d, ignoring.\n", value, lineN);
        }

        free(line);
        line = NULL;
        len = 0;
        saveptr = NULL;
    }
    free(line);
    fclose(logFile);

    return stats;
}

void printStats(Stats stats)
{
    printf("Number of:\n");
    printf("Requests:%lld\tResponses:%lld\tOf which errors:%lld\n", stats.nRequests, stats.nResponse, stats.nErrors);
    printf("Connections:%lld\tMaximum Simultaneous Connections:%lld\n", stats.connN, stats.maxConn);

    printf("\nType of requests:\n");
    printf("Read:%lld Write:%lld Open:%lld Close:%lld Lock:%lld Unlock:%lld\n", stats.readN, stats.writeN, stats.openN, stats.closeN, stats.lockN, stats.unlockN);
    printf("Total size of Reads:%lld\tWrites:%lld\n", stats.readSize, stats.writeSize);
    if (stats.readN != 0)
    {
        printf("Average size of Reads:%lld\n", stats.readSize / stats.readN);
    }
    if (stats.writeN != 0)
    {
        printf("Average size of Writes:%lld\n", stats.writeSize / stats.writeN);
    }

    printf("\nSize of server:\n");
    printf("Maximum size:%lld\n", stats.maxSize);
    printf("Maximum number of files:%lld\n", stats.maxN);
    printf("Space saved through compression:%lld\n", stats.spaceSaved);
    printf("Number of capacity misses:%lld\tNumber of files removed to free space:%lld\n", stats.missN, stats.missRemoved);
}
