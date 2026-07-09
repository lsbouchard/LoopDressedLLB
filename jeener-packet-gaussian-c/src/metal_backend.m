#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "jpg_metal.h"

#include <stdio.h>

static void set_message(char *message, size_t message_len, NSString *text) {
    if (!message || message_len == 0) return;
    const char *utf8 = text ? [text UTF8String] : "";
    snprintf(message, message_len, "%s", utf8 ? utf8 : "");
}

int jpg_metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

const char *jpg_metal_backend_name(void) {
    return "metal";
}

int jpg_metal_compile_shader_file(const char *source_path, char *message, size_t message_len) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            set_message(message, message_len, @"Metal device not available");
            return 0;
        }

        NSString *path = source_path && source_path[0]
            ? [NSString stringWithUTF8String:source_path]
            : @"shaders/jeener_kernels.metal";
        NSError *read_error = nil;
        NSString *source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&read_error];
        if (!source) {
            NSString *detail = read_error ? [read_error localizedDescription] : @"unknown read error";
            set_message(message, message_len,
                        [NSString stringWithFormat:@"could not read Metal shader source %@: %@", path, detail]);
            return 0;
        }

        NSError *compile_error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:options
                                                        error:&compile_error];
        if (!library) {
            NSString *detail = compile_error ? [compile_error localizedDescription] : @"unknown compile error";
            set_message(message, message_len,
                        [NSString stringWithFormat:@"could not compile Metal shader source %@: %@", path, detail]);
            return 0;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"matmul16_kernel"];
        if (!fn) {
            set_message(message, message_len, @"compiled Metal library does not contain matmul16_kernel");
            return 0;
        }

        set_message(message, message_len,
                    [NSString stringWithFormat:@"runtime-compiled Metal shader source %@", path]);
        return 1;
    }
}
