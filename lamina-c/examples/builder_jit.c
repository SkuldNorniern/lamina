/*
 * builder_jit.c — build an add(i64, i64) function and run it via JIT.
 *
 * Requires lamina-c built with the nightly feature:
 *   cargo build -p lamina-c --features nightly
 *
 * Build:
 *   cc builder_jit.c \
 *       -I../include \
 *       -I../include/nightly \
 *       -L../../target/debug \
 *       -llamina_c \
 *       -o builder_jit
 *
 * Run:
 *   LD_LIBRARY_PATH=../../target/debug ./builder_jit
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "lamina.h"
#include "lamina_nightly.h"

static void die(const char *where) {
    const char *err = lia_last_error();
    fprintf(stderr, "%s failed: %s\n", where, err ? err : "(no error message)");
    exit(1);
}

int main(void) {
    printf("lamina-c ABI  : %s\n", lia_version());
    printf("lamina        : %s\n", lia_compiler_version());
    printf("host target    : %s\n", lia_host_target());

    /* --- Build add(i64, i64) -> i64 ------------------------------------- */
    lia_builder_t *b = lia_builder_create();
    if (!b) { fprintf(stderr, "lia_builder_create returned NULL\n"); return 1; }

    lia_type_t *i64 = lia_type_i64();

    lia_param_t params[2];
    params[0].name = "a"; params[0].ty = i64;
    params[1].name = "b"; params[1].ty = i64;

    if (lia_builder_function(b, "add", params, 2, i64) != LIA_OK)
        die("lia_builder_function");

    lia_value_t *va = lia_value_var("a");
    lia_value_t *vb = lia_value_var("b");
    if (lia_builder_binary(b, LIA_BIN_ADD, "result", i64, va, vb) != LIA_OK)
        die("lia_builder_binary");

    lia_value_t *vr = lia_value_var("result");
    if (lia_builder_return(b, i64, vr) != LIA_OK)
        die("lia_builder_return");

    lia_value_free(va);
    lia_value_free(vb);
    lia_value_free(vr);
    lia_type_free(i64);

    lia_module_t *mod = NULL;
    if (lia_builder_finish(b, &mod) != LIA_OK)
        die("lia_builder_finish");
    lia_builder_free(b);

    /* --- JIT compile ----------------------------------------------------- */
    lia_jit_t *jit = NULL;
    if (lia_module_compile_jit(mod, "add", &jit) != LIA_OK)
        die("lia_module_compile_jit");
    lia_module_free(mod);

    /* --- Call the JIT function ------------------------------------------- */
    int64_t result = 0;
    if (lia_jit_call_i64_2(jit, 10, 32, &result) != LIA_OK)
        die("lia_jit_call_i64_2");

    printf("add(10, 32) = %lld\n", (long long)result);
    assert(result == 42);

    /* --- Cleanup --------------------------------------------------------- */
    lia_jit_free(jit);
    /* function pointer is now invalid — do not call after this point */

    printf("Done.\n");
    return 0;
}
