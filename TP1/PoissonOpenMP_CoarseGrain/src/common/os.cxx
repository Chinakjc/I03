
#include <sys/types.h>
#include <cstring>
#include <cstdlib>
#include "os.hxx"

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

#define PATH_MAX_STRING_SIZE 256

int mkdir_portable(const char *path)
{
#if defined(_WIN32)
  return _mkdir(path);
#else 
  return mkdir(path, 0755);
#endif
}

/* recursive mkdir */
int mkdir_p(const char *dir) {
  
    char tmp[PATH_MAX_STRING_SIZE];
    char *p = NULL;
    struct stat sb;
    size_t len;
    
    /* copy path */
    len = strnlen (dir, PATH_MAX_STRING_SIZE);
    if (len == 0 || len == PATH_MAX_STRING_SIZE) {
        return -1;
    }
    memcpy (tmp, dir, len);
    tmp[len] = '\0';

    /* remove trailing slash */
    if(tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    /* check if path exists and is a directory */
    if (stat (tmp, &sb) == 0) {
        if (sb.st_mode & S_IFDIR) {
            return 0;
        }
    }
    
    /* recursive mkdir */
    for(p = tmp + 1; *p; p++) {
        if(*p == '/') {
            *p = 0;
            /* test path */
            if (stat(tmp, &sb) != 0) {
                /* path does not exist - create directory */
                if (mkdir_portable(tmp) < 0) {
                    return -1;
                }
            }
            else if (!(sb.st_mode & S_IFDIR)) {
                /* not a directory */
                return -1;
            }
            *p = '/';
        }
    }
    /* test path */
    if (stat(tmp, &sb) != 0) {
        /* path does not exist - create directory */
        if (mkdir_portable(tmp) < 0) {
            return -1;
        }
        else if (!(sb.st_mode & S_IFDIR)) {
        /* not a directory */
        return -1;
        }
    }
    return 0;
}
