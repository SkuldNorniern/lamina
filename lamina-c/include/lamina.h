/*
 * lamina.h — stable C API for the Lamina compiler.
 *
 * Ownership rules:
 *   - Every handle returned by a constructor must be freed with the
 *     matching *_free function.
 *   - lia_buffer_t must be freed with lia_buffer_free().
 *   - Builder calls clone all strings and value handles; temporary handles
 *     may be freed immediately after each call.
 *   - lia_last_error() is valid only until the next API call on this thread.
 *
 * NOTE: This header is maintained by hand. Auto-generation via cbindgen is
 * pending upstream support for #[unsafe(no_mangle)] (Rust 2024 edition).
 */

#ifndef LIA_H
#define LIA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LIA_OK                     = 0,
    LIA_ERROR_INVALID_ARGUMENT = 1,
    LIA_ERROR_PARSE            = 2,
    LIA_ERROR_VALIDATION       = 3,
    LIA_ERROR_CODEGEN          = 4,
    LIA_ERROR_IO               = 5,
    LIA_ERROR_INTERNAL         = 6
} lia_status_t;

typedef struct lia_builder lia_builder_t;
typedef struct lia_module  lia_module_t;
typedef struct lia_type    lia_type_t;
typedef struct lia_value   lia_value_t;

typedef struct { uint8_t *data; size_t len; } lia_buffer_t;

typedef struct {
    const char *target;
    size_t      codegen_units;
    uint8_t     opt_level;
} lia_compile_options_t;

typedef struct {
    const char          *name;
    const lia_type_t *ty;
} lia_param_t;

typedef enum {
    LIA_BIN_ADD=0, LIA_BIN_SUB=1, LIA_BIN_MUL=2, LIA_BIN_DIV=3,
    LIA_BIN_REM=4, LIA_BIN_AND=5, LIA_BIN_OR=6,  LIA_BIN_XOR=7,
    LIA_BIN_SHL=8, LIA_BIN_SHR=9
} lia_binary_op_t;

typedef enum {
    LIA_CMP_EQ=0, LIA_CMP_NE=1, LIA_CMP_GT=2,
    LIA_CMP_GE=3, LIA_CMP_LT=4, LIA_CMP_LE=5
} lia_cmp_op_t;

const char *lia_version(void);
const char *lia_host_target(void);
const char *lia_last_error(void);
void        lia_clear_error(void);
void        lia_buffer_free(lia_buffer_t *buf);

lia_type_t *lia_type_void(void);
lia_type_t *lia_type_i8(void);   lia_type_t *lia_type_i16(void);
lia_type_t *lia_type_i32(void);  lia_type_t *lia_type_i64(void);
lia_type_t *lia_type_u8(void);   lia_type_t *lia_type_u16(void);
lia_type_t *lia_type_u32(void);  lia_type_t *lia_type_u64(void);
lia_type_t *lia_type_f32(void);  lia_type_t *lia_type_f64(void);
lia_type_t *lia_type_bool(void); lia_type_t *lia_type_ptr(void);
void           lia_type_free(lia_type_t *ty);

lia_value_t *lia_value_var(const char *name);
lia_value_t *lia_value_global(const char *name);
lia_value_t *lia_value_i8(int8_t v);    lia_value_t *lia_value_i16(int16_t v);
lia_value_t *lia_value_i32(int32_t v);  lia_value_t *lia_value_i64(int64_t v);
lia_value_t *lia_value_u8(uint8_t v);   lia_value_t *lia_value_u16(uint16_t v);
lia_value_t *lia_value_u32(uint32_t v);
lia_value_t *lia_value_u64(uint64_t v); lia_value_t *lia_value_f32(float v);
lia_value_t *lia_value_f64(double v);   lia_value_t *lia_value_bool(bool v);
lia_value_t *lia_value_string(const char *s);
void            lia_value_free(lia_value_t *val);

lia_builder_t *lia_builder_create(void);
void              lia_builder_free(lia_builder_t *b);

lia_status_t lia_builder_function(lia_builder_t *b, const char *name,
    const lia_param_t *params, size_t n, const lia_type_t *ret);
lia_status_t lia_builder_external_function(lia_builder_t *b, const char *name,
    const lia_param_t *params, size_t n, const lia_type_t *ret);
lia_status_t lia_builder_block(lia_builder_t *b, const char *name);
lia_status_t lia_builder_set_entry_block(lia_builder_t *b, const char *name);
lia_status_t lia_builder_finish(lia_builder_t *b, lia_module_t **out);

lia_status_t lia_builder_binary(lia_builder_t *b, lia_binary_op_t op,
    const char *result, const lia_type_t *ty,
    const lia_value_t *lhs, const lia_value_t *rhs);
lia_status_t lia_builder_cmp(lia_builder_t *b, lia_cmp_op_t op,
    const char *result, const lia_type_t *ty,
    const lia_value_t *lhs, const lia_value_t *rhs);
lia_status_t lia_builder_branch(lia_builder_t *b,
    const lia_value_t *cond, const char *t, const char *f);
lia_status_t lia_builder_jump(lia_builder_t *b, const char *target);
lia_status_t lia_builder_call(lia_builder_t *b, const char *result,
    const char *fn_name, const lia_value_t *const *args, size_t n);
lia_status_t lia_builder_phi(lia_builder_t *b, const char *result,
    const lia_type_t *ty, const lia_value_t *const *vals,
    const char *const *labels, size_t n);
lia_status_t lia_builder_return(lia_builder_t *b,
    const lia_type_t *ty, const lia_value_t *val);
lia_status_t lia_builder_return_void(lia_builder_t *b);
lia_status_t lia_builder_alloc_stack(lia_builder_t *b,
    const char *result, const lia_type_t *ty);
lia_status_t lia_builder_alloc_heap(lia_builder_t *b,
    const char *result, const lia_type_t *ty);
lia_status_t lia_builder_load(lia_builder_t *b, const char *result,
    const lia_type_t *ty, const lia_value_t *ptr);
lia_status_t lia_builder_store(lia_builder_t *b,
    const lia_type_t *ty, const lia_value_t *ptr, const lia_value_t *val);
lia_status_t lia_builder_dealloc(lia_builder_t *b, const lia_value_t *ptr);
lia_status_t lia_builder_getelementptr(lia_builder_t *b, const char *result,
    const lia_value_t *arr, const lia_value_t *idx, const lia_type_t *elem);
lia_status_t lia_builder_struct_gep(lia_builder_t *b, const char *result,
    const lia_value_t *ptr, size_t field);
lia_status_t lia_builder_ptrtoint(lia_builder_t *b, const char *result,
    const lia_value_t *ptr, const lia_type_t *tgt);
lia_status_t lia_builder_inttoptr(lia_builder_t *b, const char *result,
    const lia_value_t *val, const lia_type_t *tgt);
lia_status_t lia_builder_zext(lia_builder_t *b, const char *result,
    const lia_type_t *src, const lia_type_t *tgt, const lia_value_t *val);
lia_status_t lia_builder_sext(lia_builder_t *b, const char *result,
    const lia_type_t *src, const lia_type_t *tgt, const lia_value_t *val);
lia_status_t lia_builder_trunc(lia_builder_t *b, const char *result,
    const lia_type_t *src, const lia_type_t *tgt, const lia_value_t *val);
lia_status_t lia_builder_bitcast(lia_builder_t *b, const char *result,
    const lia_type_t *src, const lia_type_t *tgt, const lia_value_t *val);
lia_status_t lia_builder_select(lia_builder_t *b, const char *result,
    const lia_type_t *ty, const lia_value_t *cond,
    const lia_value_t *tv, const lia_value_t *fv);
lia_status_t lia_builder_write(lia_builder_t *b, const char *result,
    const lia_value_t *buf, const lia_value_t *size);
lia_status_t lia_builder_read(lia_builder_t *b, const char *result,
    const lia_value_t *buf, const lia_value_t *size);
lia_status_t lia_builder_write_byte(lia_builder_t *b, const char *result,
    const lia_value_t *val);
lia_status_t lia_builder_read_byte(lia_builder_t *b, const char *result);
lia_status_t lia_builder_write_ptr(lia_builder_t *b, const char *result,
    const lia_value_t *ptr);

lia_status_t lia_module_emit_ir(const lia_module_t *m, lia_buffer_t *out);
lia_status_t lia_compile_ir_to_assembly(const char *ir,
    const lia_compile_options_t *opts, lia_buffer_t *out);
lia_status_t lia_module_compile_to_assembly(const lia_module_t *m,
    const lia_compile_options_t *opts, lia_buffer_t *out);
void lia_module_free(lia_module_t *m);

#ifdef __cplusplus
}
#endif

#endif /* LIA_H */
