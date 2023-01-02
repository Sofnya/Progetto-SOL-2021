#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

#include "SERVER/logging.h"
#include "SERVER/globals.h"
#include "COMMON/macros.h"
#include "COMMON/helpers.h"

volatile pthread_mutex_t LOGLOCK;
/**
 * @brief Logs a message, of given type. Type should be machine readable.
 *
 * @param msg the message to log.
 * @param type the type of the message.
 * @return int 0 on success, -1 and sets errno on failure.
 */
int logger(char *msg, char *type)
{
    FILE *file;
    time_t tmpTime;
    struct tm *tm;
    char *parsed;

    time(&tmpTime);
    tm = gmtime(&tmpTime);

    SAFE_NULL_CHECK(parsed = malloc(strlen(msg) + strlen(type) + 500));

    // Where the magical formatting happens.
    sprintf(parsed, "%02d/%02d/%04d %02d:%02d:%02d\tTID:%ld\t[%s: %s ]\n", tm->tm_mday, tm->tm_mon + 1, tm->tm_year + 1900, tm->tm_hour, tm->tm_min, tm->tm_sec, getTID(), type, msg);

    // We only write our output while holding a mutex to avoid concurrency problems.
    PTHREAD_CHECK(pthread_mutex_lock((pthread_mutex_t *)&LOGLOCK));
    CLEANUP_CHECK(file = fopen(LOG_FILE, "a"), NULL, pthread_mutex_unlock((pthread_mutex_t *)&LOGLOCK));

    fwrite(parsed, strlen(parsed), 1, file);
    fclose(file);
    // We can mirror the logging to stdout if needed.
    if (VERBOSE_PRINT)
    {
        printf("%s", parsed);
    }

    PTHREAD_CHECK(pthread_mutex_unlock((pthread_mutex_t *)&LOGLOCK));

    free(parsed);

    return 0;
}