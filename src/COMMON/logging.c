#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "COMMON/logging.h"
#include "SERVER/globals.h"
#include "COMMON/macros.h"

#define LOG_FILE "log"

int logger(char *msg)
{
    FILE *file;
    time_t tmpTime;
    struct tm *tm;
    char *parsed;

    time(&tmpTime);
    tm = gmtime(&tmpTime);
    SAFE_NULL_CHECK(file = fopen(LOG_FILE, "a"));

    SAFE_NULL_CHECK(parsed = malloc(strlen(msg) + 500));

    sprintf(parsed, "%d/%d/%d %d:%d:%d\t[ %s ]\n", tm->tm_mday, tm->tm_mon, tm->tm_year + 1900, tm->tm_hour, tm->tm_min, tm->tm_sec, msg);
    fwrite(parsed, strlen(parsed), 1, file);
    fflush(file);
    printf("%s", parsed);
    fclose(file);
    free(parsed);

    return 0;
}