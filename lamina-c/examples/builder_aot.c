/*
 * builder_aot.c — build an add(i64, i64) function via the C builder API
 * and compile it to assembly using AOT.
 *
 * Build (after cargo build -p lamina-c):
 *
 *   cc builder_aot.c \
 *       -I../include \
 *       -L../../target/debug \
 *       -llamina_c \
 *       -o builder_aot
 *
 * Run:
 *   LD_LIBRARY_PATH=../../target/debug ./builder_aot
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lamina.h"

static void die(const char *where) {
    const char *err = lamina_last_error();
    fprintf(stderr, "%s failed: %s\n", where, err ? err : "(no error message)");
    exit(1);
}

int main(void) {
    printf("lamina version : %s\n", lamina_version());
    printf("host target    : %s\n", lamina_host_target());

    /* --- Create builder ------------------------------------------------- */
    lamina_builder_t *b = lamina_builder_create();
    if (!b) { fprintf(stderr, "lamina_builder_create returned NULL\n"); return 1; }

    /* --- Define add(i64 %a, i64 %b) -> i64 ------------------------------ */
    lamina_type_t *i64 = lamina_type_i64();

    lamina_param_t params[2];
    params[0].name = "a";
    params[0].ty   = i64;
    params[1].name = "b";
    params[1].ty   = i64;

    if (lamina_builder_function(b, "add", params, 2, i64) != LAMINA_OK)
        die("lamina_builder_function");

    /* entry block is created automatically */

    /* %result = add.i64 %a, %b */
    lamina_value_t *va = lamina_value_var("a");
    lamina_value_t *vb = lamina_value_var("b");
    if (lamina_builder_binary(b, LAMINA_BIN_ADD, "result", i64, va, vb) != LAMINA_OK)
        die("lamina_builder_binary");

    /* ret.i64 %result */
    lamina_value_t *vr = lamina_value_var("result");
    if (lamina_builder_return(b, i64, vr) != LAMINA_OK)
        die("lamina_builder_return");

    /* --- Free temporaries (safe to do before finish) -------------------- */
    lamina_value_free(va);
    lamina_value_free(vb);
    lamina_value_free(vr);
    lamina_type_free(i64);

    /* --- Finish builder → module ---------------------------------------- */
    lamina_module_t *mod = NULL;
    if (lamina_builder_finish(b, &mod) != LAMINA_OK)
        die("lamina_builder_finish");
    lamina_builder_free(b);

    /* --- Emit IR text ---------------------------------------------------- */
    lamina_buffer_t ir_buf = {0};
    if (lamina_module_emit_ir(mod, &ir_buf) != LAMINA_OK)
        die("lamina_module_emit_ir");

    printf("\n--- Generated Lamina IR ---\n%.*s\n", (int)ir_buf.len, (char *)ir_buf.data);
    lamina_buffer_free(&ir_buf);

    /* --- Compile to assembly (host target, defaults) -------------------- */
    lamina_buffer_t asm_buf = {0};
    if (lamina_module_compile_to_assembly(mod, NULL, &asm_buf) != LAMINA_OK)
        die("lamina_module_compile_to_assembly");

    printf("--- Assembly (%zu bytes) ---\n%.*s\n",
           asm_buf.len, (int)asm_buf.len, (char *)asm_buf.data);

    lamina_buffer_free(&asm_buf);
    lamina_module_free(mod);

    printf("Done.\n");
    return 0;
}
