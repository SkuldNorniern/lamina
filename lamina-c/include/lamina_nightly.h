/*
 * lamina_nightly.h — nightly JIT extension to the Lamina C API.
 *
 * Requires lamina-c built with --features nightly.
 *
 * WARNING: No stable ABI guarantee. These declarations may change between
 * releases without notice. Do not use in production code.
 *
 * Function pointers obtained from lia_jit_get_function() become invalid
 * the moment lia_jit_free() is called on the owning handle.
 */

#ifndef LIA_NIGHTLY_H
#define LIA_NIGHTLY_H

#include "lamina.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * JIT handle (nightly only)
 * ---------------------------------------------------------------------- */

typedef struct lia_jit lia_jit_t;

/* -------------------------------------------------------------------------
 * JIT compilation
 * ---------------------------------------------------------------------- */

lia_status_t lia_jit_compile_ir(
    const char   *ir,
    const char   *function_name,
    lia_jit_t **jit_out);

lia_status_t lia_module_compile_jit(
    const lia_module_t *module,
    const char            *function_name,
    lia_jit_t         **jit_out);

lia_status_t lia_jit_get_function(
    const lia_jit_t *jit,
    void              **function_out);

void lia_jit_free(lia_jit_t *jit);

/* -------------------------------------------------------------------------
 * Typed call helpers
 * ---------------------------------------------------------------------- */

lia_status_t lia_jit_call_i64_0(
    lia_jit_t *jit,
    int64_t      *result);

lia_status_t lia_jit_call_i64_1(
    lia_jit_t *jit,
    int64_t       a,
    int64_t      *result);

lia_status_t lia_jit_call_i64_2(
    lia_jit_t *jit,
    int64_t       a,
    int64_t       b,
    int64_t      *result);

lia_status_t lia_jit_call_i64_3(
    lia_jit_t *jit,
    int64_t       a,
    int64_t       b,
    int64_t       c,
    int64_t      *result);

lia_status_t lia_jit_call_i64_4(
    lia_jit_t *jit,
    int64_t       a,
    int64_t       b,
    int64_t       c,
    int64_t       d,
    int64_t      *result);

#ifdef __cplusplus
}
#endif

#endif /* LIA_NIGHTLY_H */
