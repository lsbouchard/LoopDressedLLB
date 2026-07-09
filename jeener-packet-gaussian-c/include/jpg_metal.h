#ifndef JPG_METAL_H
#define JPG_METAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int jpg_metal_available(void);
const char *jpg_metal_backend_name(void);
int jpg_metal_compile_shader_file(const char *source_path, char *message, size_t message_len);

#ifdef __cplusplus
}
#endif

#endif
