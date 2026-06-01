/*
 * lamina_nightly.h — nightly JIT extension to the Lamina C API.
 *
 * Requires lamina-c built with --features nightly.
 *
 * WARNING: No stable ABI guarantee. These declarations may change between
 * releases without notice. Do not use in production code.
 *
 * Function pointers obtained from lamina_jit_get_function() become invalid
 * the moment lamina_jit_free() is called on the owning handle.
 */

#ifndef LAMINA_NIGHTLY_H
#define LAMINA_NIGHTLY_H

#include "lamina.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * JIT handle (nightly only)
 * ---------------------------------------------------------------------- */

typedef struct lamina_jit lamina_jit_t;

/* -------------------------------------------------------------------------
 * JIT compilation
 * ---------------------------------------------------------------------- */

lamina_status_t lamina_jit_compile_ir(
    const char   *ir,
    const char   *function_name,
    lamina_jit_t **jit_out);

lamina_status_t lamina_module_compile_jit(
    const lamina_module_t *module,
    const char            *function_name,
    lamina_jit_t         **jit_out);

lamina_status_t lamina_jit_get_function(
    const lamina_jit_t *jit,
    void              **function_out);

void lamina_jit_free(lamina_jit_t *jit);

/* -------------------------------------------------------------------------
 * Typed call helpers
 * ---------------------------------------------------------------------- */

lamina_status_t lamina_jit_call_i64_0(
    lamina_jit_t *jit,
    int64_t      *result);

lamina_status_t lamina_jit_call_i64_1(
    lamina_jit_t *jit,
    int64_t       a,
    int64_t      *result);

lamina_status_t lamina_jit_call_i64_2(
    lamina_jit_t *jit,
    int64_t       a,
    int64_t       b,
    int64_t      *result);

#ifdef __cplusplus
}
#endif

#endif /* LAMINA_NIGHTLY_H */
