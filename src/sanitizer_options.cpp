/*
Default runtime options for the sanitizer build (NTRACER_SANITIZE=ON).
Only compiled into that build.
*/

// An oversized calloc must return NULL as it does in the normal build (the
// read route turns that into a 500) rather than abort the process.
extern "C" const char *__asan_default_options()
{
    return "allocator_may_return_null=1";
}

extern "C" const char *__ubsan_default_options()
{
    return "print_stacktrace=1";
}
