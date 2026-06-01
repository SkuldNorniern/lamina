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
    const char *err = lamina_last_error();
    fprintf(stderr, "%s failed: %s\n", where, err ? err : "(no error message)");
    exit(1);
}

int main(void) {
    printf("lamina version : %s\n", lamina_version());
    printf("host target    : %s\n", lamina_host_target());

    /* --- Build add(i64, i64) -> i64 ------------------------------------- */
    lamina_builder_t *b = lamina_builder_create();
    if (!b) { fprintf(stderr, "lamina_builder_create returned NULL\n"); return 1; }

    lamina_type_t *i64 = lamina_type_i64();

    lamina_param_t params[2];
    params[0].name = "a"; params[0].ty = i64;
    params[1].name = "b"; params[1].ty = i64;

    if (lamina_builder_function(b, "add", params, 2, i64) != LAMINA_OK)
        die("lamina_builder_function");

    lamina_value_t *va = lamina_value_var("a");
    lamina_value_t *vb = lamina_value_var("b");
    if (lamina_builder_binary(b, LAMINA_BIN_ADD, "result", i64, va, vb) != LAMINA_OK)
        die("lamina_builder_binary");

    lamina_value_t *vr = lamina_value_var("result");
    if (lamina_builder_return(b, i64, vr) != LAMINA_OK)
        die("lamina_builder_return");

    lamina_value_free(va);
    lamina_value_free(vb);
    lamina_value_free(vr);
    lamina_type_free(i64);

    lamina_module_t *mod = NULL;
    if (lamina_builder_finish(b, &mod) != LAMINA_OK)
        die("lamina_builder_finish");
    lamina_builder_free(b);

    /* --- JIT compile ----------------------------------------------------- */
    lamina_jit_t *jit = NULL;
    if (lamina_module_compile_jit(mod, "add", &jit) != LAMINA_OK)
        die("lamina_module_compile_jit");
    lamina_module_free(mod);

    /* --- Call the JIT function ------------------------------------------- */
    int64_t result = 0;
    if (lamina_jit_call_i64_2(jit, 10, 32, &result) != LAMINA_OK)
        die("lamina_jit_call_i64_2");

    printf("add(10, 32) = %lld\n", (long long)result);
    assert(result == 42);

    /* --- Cleanup --------------------------------------------------------- */
    lamina_jit_free(jit);
    /* function pointer is now invalid — do not call after this point */

    printf("Done.\n");
    return 0;
}
