#include "jpg_metal.h"

#include <stdio.h>

int jpg_metal_available(void) {
    return 0;
}

const char *jpg_metal_backend_name(void) {
    return "cpu";
}

int jpg_metal_compile_shader_file(const char *source_path, char *message, size_t message_len) {
    (void)source_path;
    if (message && message_len > 0) {
        snprintf(message, message_len, "%s", "Metal runtime shader compilation is unavailable on this platform");
    }
    return 0;
}
