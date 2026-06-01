/*
 * lamina.h — stable C API for the Lamina compiler.
 *
 * Ownership rules:
 *   - Every handle returned by a constructor must be freed with the
 *     matching _free function.
 *   - lamina_buffer_t returned by the API must be freed with
 *     lamina_buffer_free().
 *   - Builder functions clone all values and strings passed to them.
 *     Temporary handles may be freed immediately after the call.
 *   - lamina_last_error() returns a pointer valid only until the next
 *     Lamina API call on the same thread.
 */

#ifndef LAMINA_H
#define LAMINA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Status codes
 * ---------------------------------------------------------------------- */

typedef enum {
    LAMINA_OK                    = 0,
    LAMINA_ERROR_INVALID_ARGUMENT = 1,
    LAMINA_ERROR_PARSE           = 2,
    LAMINA_ERROR_VALIDATION      = 3,
    LAMINA_ERROR_CODEGEN         = 4,
    LAMINA_ERROR_IO              = 5,
    LAMINA_ERROR_INTERNAL        = 6
} lamina_status_t;

/* -------------------------------------------------------------------------
 * Opaque handle types
 * ---------------------------------------------------------------------- */

typedef struct lamina_builder lamina_builder_t;
typedef struct lamina_module  lamina_module_t;
typedef struct lamina_type    lamina_type_t;
typedef struct lamina_value   lamina_value_t;

/* -------------------------------------------------------------------------
 * Buffer (heap-allocated byte array)
 * ---------------------------------------------------------------------- */

typedef struct {
    uint8_t *data;
    size_t   len;
} lamina_buffer_t;

/* -------------------------------------------------------------------------
 * Compile options
 *
 * target        — target identifier string, e.g. "x86_64_linux".
 *                 NULL selects the host target.
 * codegen_units — parallel compilation units (0 treated as 1).
 * opt_level     — reserved; must be 0.
 * ---------------------------------------------------------------------- */

typedef struct {
    const char *target;
    size_t      codegen_units;
    uint8_t     opt_level;
} lamina_compile_options_t;

/* -------------------------------------------------------------------------
 * Function parameter (passed by value to builder calls)
 * ---------------------------------------------------------------------- */

typedef struct {
    const char          *name;
    const lamina_type_t *ty;
} lamina_param_t;

/* -------------------------------------------------------------------------
 * Binary and comparison op enums
 * ---------------------------------------------------------------------- */

typedef enum {
    LAMINA_BIN_ADD = 0,
    LAMINA_BIN_SUB = 1,
    LAMINA_BIN_MUL = 2,
    LAMINA_BIN_DIV = 3,
    LAMINA_BIN_REM = 4,
    LAMINA_BIN_AND = 5,
    LAMINA_BIN_OR  = 6,
    LAMINA_BIN_XOR = 7,
    LAMINA_BIN_SHL = 8,
    LAMINA_BIN_SHR = 9
} lamina_binary_op_t;

typedef enum {
    LAMINA_CMP_EQ = 0,
    LAMINA_CMP_NE = 1,
    LAMINA_CMP_GT = 2,
    LAMINA_CMP_GE = 3,
    LAMINA_CMP_LT = 4,
    LAMINA_CMP_LE = 5
} lamina_cmp_op_t;

/* -------------------------------------------------------------------------
 * Version / info
 * ---------------------------------------------------------------------- */

const char *lamina_version(void);
const char *lamina_host_target(void);

/* -------------------------------------------------------------------------
 * Error handling
 * ---------------------------------------------------------------------- */

const char *lamina_last_error(void);
void        lamina_clear_error(void);

/* -------------------------------------------------------------------------
 * Buffer
 * ---------------------------------------------------------------------- */

void lamina_buffer_free(lamina_buffer_t *buf);

/* -------------------------------------------------------------------------
 * Type constructors
 * ---------------------------------------------------------------------- */

lamina_type_t *lamina_type_void(void);
lamina_type_t *lamina_type_i8(void);
lamina_type_t *lamina_type_i16(void);
lamina_type_t *lamina_type_i32(void);
lamina_type_t *lamina_type_i64(void);
lamina_type_t *lamina_type_u8(void);
lamina_type_t *lamina_type_u16(void);
lamina_type_t *lamina_type_u32(void);
lamina_type_t *lamina_type_u64(void);
lamina_type_t *lamina_type_f32(void);
lamina_type_t *lamina_type_f64(void);
lamina_type_t *lamina_type_bool(void);
lamina_type_t *lamina_type_ptr(void);
void           lamina_type_free(lamina_type_t *ty);

/* -------------------------------------------------------------------------
 * Value constructors
 * ---------------------------------------------------------------------- */

lamina_value_t *lamina_value_var(const char *name);
lamina_value_t *lamina_value_global(const char *name);
lamina_value_t *lamina_value_i8(int8_t v);
lamina_value_t *lamina_value_i16(int16_t v);
lamina_value_t *lamina_value_i32(int32_t v);
lamina_value_t *lamina_value_i64(int64_t v);
lamina_value_t *lamina_value_u8(uint8_t v);
lamina_value_t *lamina_value_u32(uint32_t v);
lamina_value_t *lamina_value_u64(uint64_t v);
lamina_value_t *lamina_value_f32(float v);
lamina_value_t *lamina_value_f64(double v);
lamina_value_t *lamina_value_bool(bool v);
lamina_value_t *lamina_value_string(const char *s);
void            lamina_value_free(lamina_value_t *val);

/* -------------------------------------------------------------------------
 * Builder lifecycle
 * ---------------------------------------------------------------------- */

lamina_builder_t *lamina_builder_create(void);
void              lamina_builder_free(lamina_builder_t *builder);

/* -------------------------------------------------------------------------
 * Function / block definition
 * ---------------------------------------------------------------------- */

lamina_status_t lamina_builder_function(
    lamina_builder_t       *builder,
    const char             *name,
    const lamina_param_t   *params,
    size_t                  param_count,
    const lamina_type_t    *return_type);

lamina_status_t lamina_builder_external_function(
    lamina_builder_t       *builder,
    const char             *name,
    const lamina_param_t   *params,
    size_t                  param_count,
    const lamina_type_t    *return_type);

lamina_status_t lamina_builder_block(
    lamina_builder_t *builder,
    const char       *name);

lamina_status_t lamina_builder_set_entry_block(
    lamina_builder_t *builder,
    const char       *name);

lamina_status_t lamina_builder_finish(
    lamina_builder_t  *builder,
    lamina_module_t  **module_out);

/* -------------------------------------------------------------------------
 * Instructions
 * ---------------------------------------------------------------------- */

lamina_status_t lamina_builder_binary(
    lamina_builder_t      *builder,
    lamina_binary_op_t     op,
    const char            *result,
    const lamina_type_t   *ty,
    const lamina_value_t  *lhs,
    const lamina_value_t  *rhs);

lamina_status_t lamina_builder_cmp(
    lamina_builder_t      *builder,
    lamina_cmp_op_t        op,
    const char            *result,
    const lamina_type_t   *ty,
    const lamina_value_t  *lhs,
    const lamina_value_t  *rhs);

lamina_status_t lamina_builder_branch(
    lamina_builder_t     *builder,
    const lamina_value_t *condition,
    const char           *true_label,
    const char           *false_label);

lamina_status_t lamina_builder_jump(
    lamina_builder_t *builder,
    const char       *target);

lamina_status_t lamina_builder_call(
    lamina_builder_t            *builder,
    const char                  *result,
    const char                  *func_name,
    const lamina_value_t *const *args,
    size_t                       arg_count);

lamina_status_t lamina_builder_phi(
    lamina_builder_t            *builder,
    const char                  *result,
    const lamina_type_t         *ty,
    const lamina_value_t *const *values,
    const char          *const  *labels,
    size_t                       count);

lamina_status_t lamina_builder_return(
    lamina_builder_t     *builder,
    const lamina_type_t  *ty,
    const lamina_value_t *value);

lamina_status_t lamina_builder_return_void(lamina_builder_t *builder);

lamina_status_t lamina_builder_alloc_stack(
    lamina_builder_t    *builder,
    const char          *result,
    const lamina_type_t *ty);

lamina_status_t lamina_builder_alloc_heap(
    lamina_builder_t    *builder,
    const char          *result,
    const lamina_type_t *ty);

lamina_status_t lamina_builder_load(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_type_t  *ty,
    const lamina_value_t *ptr);

lamina_status_t lamina_builder_store(
    lamina_builder_t     *builder,
    const lamina_type_t  *ty,
    const lamina_value_t *ptr,
    const lamina_value_t *value);

lamina_status_t lamina_builder_dealloc(
    lamina_builder_t     *builder,
    const lamina_value_t *ptr);

lamina_status_t lamina_builder_getelementptr(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *array_ptr,
    const lamina_value_t *index,
    const lamina_type_t  *element_type);

lamina_status_t lamina_builder_struct_gep(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *struct_ptr,
    size_t                field_index);

lamina_status_t lamina_builder_ptrtoint(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *ptr_value,
    const lamina_type_t  *target_type);

lamina_status_t lamina_builder_inttoptr(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *int_value,
    const lamina_type_t  *target_type);

lamina_status_t lamina_builder_zext(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_type_t  *source_type,
    const lamina_type_t  *target_type,
    const lamina_value_t *value);

lamina_status_t lamina_builder_sext(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_type_t  *source_type,
    const lamina_type_t  *target_type,
    const lamina_value_t *value);

lamina_status_t lamina_builder_trunc(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_type_t  *source_type,
    const lamina_type_t  *target_type,
    const lamina_value_t *value);

lamina_status_t lamina_builder_bitcast(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_type_t  *source_type,
    const lamina_type_t  *target_type,
    const lamina_value_t *value);

lamina_status_t lamina_builder_select(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_type_t  *ty,
    const lamina_value_t *cond,
    const lamina_value_t *true_val,
    const lamina_value_t *false_val);

lamina_status_t lamina_builder_write(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *buffer,
    const lamina_value_t *size);

lamina_status_t lamina_builder_read(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *buffer,
    const lamina_value_t *size);

lamina_status_t lamina_builder_write_byte(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *value);

lamina_status_t lamina_builder_read_byte(
    lamina_builder_t *builder,
    const char       *result);

lamina_status_t lamina_builder_write_ptr(
    lamina_builder_t     *builder,
    const char           *result,
    const lamina_value_t *ptr);

/* -------------------------------------------------------------------------
 * Module
 * ---------------------------------------------------------------------- */

lamina_status_t lamina_module_emit_ir(
    const lamina_module_t *module,
    lamina_buffer_t       *output);

lamina_status_t lamina_compile_ir_to_assembly(
    const char                     *ir,
    const lamina_compile_options_t *options,
    lamina_buffer_t                *output);

lamina_status_t lamina_module_compile_to_assembly(
    const lamina_module_t          *module,
    const lamina_compile_options_t *options,
    lamina_buffer_t                *output);

void lamina_module_free(lamina_module_t *module);

#ifdef __cplusplus
}
#endif

#endif /* LAMINA_H */
