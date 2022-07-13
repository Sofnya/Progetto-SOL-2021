#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "SERVER/logging.h"
#include "SERVER/globals.h"
#include "COMMON/macros.h"

/**
 * @brief Logs a message, of type type. Type should be machine readable.
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
    SAFE_NULL_CHECK(file = fopen(LOG_FILE, "a"));

    SAFE_NULL_CHECK(parsed = malloc(strlen(msg) + strlen(type) + 500));

    sprintf(parsed, "%d/%d/%d %d:%d:%d\t[%s: %s ]\n", tm->tm_mday, tm->tm_mon, tm->tm_year + 1900, tm->tm_hour, tm->tm_min, tm->tm_sec, type, msg);
    fwrite(parsed, strlen(parsed), 1, file);
    fflush(file);

    if (VERBOSE_PRINT)
    {
        printf("%s", parsed);
    }
    fclose(file);
    free(parsed);

    return 0;
}