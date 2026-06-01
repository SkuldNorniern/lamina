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
    const char *err = lia_last_error();
    fprintf(stderr, "%s failed: %s\n", where, err ? err : "(no error message)");
    exit(1);
}

int main(void) {
    printf("lamina version : %s\n", lia_version());
    printf("host target    : %s\n", lia_host_target());

    /* --- Create builder ------------------------------------------------- */
    lia_builder_t *b = lia_builder_create();
    if (!b) { fprintf(stderr, "lia_builder_create returned NULL\n"); return 1; }

    /* --- Define add(i64 %a, i64 %b) -> i64 ------------------------------ */
    lia_type_t *i64 = lia_type_i64();

    lia_param_t params[2];
    params[0].name = "a";
    params[0].ty   = i64;
    params[1].name = "b";
    params[1].ty   = i64;

    if (lia_builder_function(b, "add", params, 2, i64) != LIA_OK)
        die("lia_builder_function");

    /* entry block is created automatically */

    /* %result = add.i64 %a, %b */
    lia_value_t *va = lia_value_var("a");
    lia_value_t *vb = lia_value_var("b");
    if (lia_builder_binary(b, LIA_BIN_ADD, "result", i64, va, vb) != LIA_OK)
        die("lia_builder_binary");

    /* ret.i64 %result */
    lia_value_t *vr = lia_value_var("result");
    if (lia_builder_return(b, i64, vr) != LIA_OK)
        die("lia_builder_return");

    /* --- Free temporaries (safe to do before finish) -------------------- */
    lia_value_free(va);
    lia_value_free(vb);
    lia_value_free(vr);
    lia_type_free(i64);

    /* --- Finish builder → module ---------------------------------------- */
    lia_module_t *mod = NULL;
    if (lia_builder_finish(b, &mod) != LIA_OK)
        die("lia_builder_finish");
    lia_builder_free(b);

    /* --- Emit IR text ---------------------------------------------------- */
    lia_buffer_t ir_buf = {0};
    if (lia_module_emit_ir(mod, &ir_buf) != LIA_OK)
        die("lia_module_emit_ir");

    printf("\n--- Generated Lamina IR ---\n%.*s\n", (int)ir_buf.len, (char *)ir_buf.data);
    lia_buffer_free(&ir_buf);

    /* --- Compile to assembly (host target, defaults) -------------------- */
    lia_buffer_t asm_buf = {0};
    if (lia_module_compile_to_assembly(mod, NULL, &asm_buf) != LIA_OK)
        die("lia_module_compile_to_assembly");

    printf("--- Assembly (%zu bytes) ---\n%.*s\n",
           asm_buf.len, (int)asm_buf.len, (char *)asm_buf.data);

    lia_buffer_free(&asm_buf);
    lia_module_free(mod);

    printf("Done.\n");
    return 0;
}
