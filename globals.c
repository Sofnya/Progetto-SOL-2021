#include "globals.h"
#include "macros.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>


char SOCK_NAME[UNIX_PATH_MAX] = "default_address";
long POOL_SIZE = 8;
long long MAX_FILES = 100;
long long MAX_MEMORY = 100*1024*1024;


void load_config(char *path)
{
    FILE *configFile;
    char *line = NULL, *option = NULL, *value = NULL, *saveptr = NULL, *endptr = NULL;
    size_t len = 0;
    int lineN = 0;
    long long tmp;

    UNSAFE_NULL_CHECK(configFile = fopen(path, "r"));

    while(getline(&line, &len, configFile) != -1)
    {
        lineN++;
        len = strlen(line);

        // Stripping of the newlines.
        if(len >= 2)
        {
            if(line[len-1] == '\n') line[len - 1] = '\00';
        }

        option = strtok_r(line, "=", &saveptr);
        value = strtok_r(NULL, "=", &saveptr);

        if((option == NULL) | (value == NULL))
        {
            printf("Malformed config file on line %d, continuing anyway.\n", lineN);
        }
        else if(!strcmp(option, "SOCK_NAME"))
        {
            strncpy(SOCK_NAME, value, UNIX_PATH_MAX -1);
            SOCK_NAME[UNIX_PATH_MAX-1] = '\00'; 
        }
        else if (!strcmp(option, "POOL_SIZE"))
        {
            tmp = strtol(value, NULL, 0);
            if(tmp == __LONG_MAX__ || tmp <= 0)
            {
                printf("Invalid value on line %d, ignoring.\n", lineN);
                continue;
            }
            else{
            POOL_SIZE = tmp;
            }
        }
        else if (!strcmp(option, "MAX_FILES"))
        {
            tmp = strtoll(value, &endptr, 0);
            if(tmp == __LONG_LONG_MAX__ || tmp <= 0)
            {
                printf("Invalid value on line %d, ignoring.\n", lineN);
            }
            else{
                MAX_FILES = tmp;

                if(!strncasecmp(endptr, "k", 1)) MAX_FILES *= 1024;
                else if (!strncasecmp(endptr, "m", 1)) MAX_FILES *= (1024*1024);
                else if (!strncasecmp(endptr, "g", 1)) MAX_FILES *= (1024*1024*1024);
            
                endptr = NULL;
            }
        }
        else if (!strcmp(option, "MAX_MEMORY"))
        {
            tmp = strtoll(value, &endptr, 0);
            if(tmp == __LONG_LONG_MAX__ || tmp <= 0)
            {
                printf("Invalid value on line %d, ignoring.\n", lineN);
            }
            else{
                MAX_MEMORY = tmp;

                if(!strncasecmp(endptr, "k", 1)) MAX_MEMORY *= 1024;
                else if (!strncasecmp(endptr, "m", 1)) MAX_MEMORY *= (1024*1024);
                else if (!strncasecmp(endptr, "g", 1)) MAX_MEMORY *= (1024*1024*1024);
                
                endptr = NULL;
                
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
    printf("SOCK_NAME:%s\nPOOL_SIZE:%ld\nMAX_FILES:%lld\nMAX_MEMORY:%lld\n", SOCK_NAME, POOL_SIZE, MAX_FILES, MAX_MEMORY);
}