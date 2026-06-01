/*
 * raw_ir.c — compile hand-written Lamina IR text to assembly via the C API.
 *
 * Shows the direct IR path (no builder). Useful when you already have IR
 * text from another source and just need AOT compilation.
 *
 * Build: see Makefile in lamina-c/
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

/* --- IR text for an add(i64, i64) -> i64 function ----------------------- */
static const char ADD_IR[] =
    "fn @add(i64 %a, i64 %b) -> i64 {\n"
    "entry:\n"
    "  %result = add.i64 %a, %b\n"
    "  ret.i64 %result\n"
    "}\n";

/* --- IR text with control flow (max of two i64s) ------------------------ */
static const char MAX_IR[] =
    "fn @max(i64 %a, i64 %b) -> i64 {\n"
    "entry:\n"
    "  %cond = gt.i64 %a, %b\n"
    "  br %cond, ret_a, ret_b\n"
    "ret_a:\n"
    "  ret.i64 %a\n"
    "ret_b:\n"
    "  ret.i64 %b\n"
    "}\n";

/* --- IR text with a local variable (stack alloc + store + load) --------- */
static const char DOUBLE_IR[] =
    "fn @double(i64 %x) -> i64 {\n"
    "entry:\n"
    "  %slot = alloc.ptr.stack i64\n"
    "  store.i64 %slot, %x\n"
    "  %val  = load.i64 %slot\n"
    "  %out  = mul.i64 %val, 2\n"
    "  ret.i64 %out\n"
    "}\n";

static void compile_and_print(const char *label, const char *ir) {
    printf("=== %s ===\n", label);
    printf("IR:\n%s\n", ir);

    lamina_buffer_t asm_buf = {0};
    if (lamina_compile_ir_to_assembly(ir, NULL, &asm_buf) != LAMINA_OK)
        die("lamina_compile_ir_to_assembly");

    printf("Assembly (%zu bytes):\n%.*s\n",
           asm_buf.len, (int)asm_buf.len, (char *)asm_buf.data);

    lamina_buffer_free(&asm_buf);
}

int main(void) {
    printf("lamina %s on %s\n\n", lamina_version(), lamina_host_target());

    compile_and_print("add(i64, i64) -> i64",        ADD_IR);
    compile_and_print("max(i64, i64) -> i64",        MAX_IR);
    compile_and_print("double(i64) -> i64 (memory)", DOUBLE_IR);

    /* --- Error path: invalid IR ----------------------------------------- */
    printf("=== invalid IR (expect error) ===\n");
    lamina_buffer_t bad_buf = {0};
    lamina_status_t st = lamina_compile_ir_to_assembly("this is not valid IR", NULL, &bad_buf);
    if (st != LAMINA_OK) {
        printf("Got expected error: %s\n\n", lamina_last_error());
    } else {
        fprintf(stderr, "Expected error for invalid IR but got LAMINA_OK\n");
        lamina_buffer_free(&bad_buf);
        return 1;
    }

    /* --- Target selection: explicit host target -------------------------- */
    printf("=== explicit target ===\n");
    lamina_compile_options_t opts = {0};
    opts.target        = lamina_host_target();
    opts.codegen_units = 1;
    opts.opt_level     = 0;

    lamina_buffer_t explicit_buf = {0};
    if (lamina_compile_ir_to_assembly(ADD_IR, &opts, &explicit_buf) != LAMINA_OK)
        die("lamina_compile_ir_to_assembly (explicit target)");

    printf("Compiled %zu bytes for target '%s'\n", explicit_buf.len, opts.target);
    lamina_buffer_free(&explicit_buf);

    printf("\nDone.\n");
    return 0;
}
