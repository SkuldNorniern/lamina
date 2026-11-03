use super::abi::{ArgSlot, aggregate_bytes, classify, frame_bytes, outgoing_bytes, slot_offset};
use crate::mir::MirType;
use lamina_mir::ScalarType;

#[test]
fn every_argument_takes_a_word_on_the_stack() {
    let types = [
        MirType::Scalar(ScalarType::I32),
        MirType::Scalar(ScalarType::I8),
        MirType::Scalar(ScalarType::Ptr),
        MirType::Scalar(ScalarType::F64),
    ];
    let places = classify(&types, &[None; 4], false);
    assert_eq!(
        places,
        [
            ArgSlot {
                offset: 0,
                bytes: 4
            },
            ArgSlot {
                offset: 4,
                bytes: 4
            },
            ArgSlot {
                offset: 8,
                bytes: 4
            },
            ArgSlot {
                offset: 12,
                bytes: 8
            },
        ]
    );
    assert_eq!(outgoing_bytes(&places, false), 32);
}

#[test]
fn a_hidden_struct_pointer_takes_the_first_word() {
    let types = [MirType::Scalar(ScalarType::I32)];
    let places = classify(&types, &[None], true);
    assert_eq!(
        places,
        [ArgSlot {
            offset: 4,
            bytes: 4
        }]
    );
    assert_eq!(outgoing_bytes(&places, true), 16);
}

#[test]
fn an_aggregate_argument_is_copied_by_value_rounded_to_a_word() {
    let types = [
        MirType::Scalar(ScalarType::Ptr),
        MirType::Scalar(ScalarType::I32),
    ];
    let places = classify(&types, &[Some(13), None], false);
    assert_eq!(
        places,
        [
            ArgSlot {
                offset: 0,
                bytes: 16
            },
            ArgSlot {
                offset: 16,
                bytes: 4
            },
        ]
    );
    assert_eq!(aggregate_bytes(13), 16);
}

#[test]
fn no_arguments_need_no_outgoing_area() {
    assert_eq!(outgoing_bytes(&[], false), 0);
}

/// `call` leaves `%esp` four past a boundary and `pushl %ebp` another four, so the
/// frame must be `8 mod 16` for `%esp` to sit on one inside the body.
#[test]
fn the_frame_restores_the_sixteen_byte_alignment_a_call_needs() {
    for locals in [0, 8, 16, 24, 120, 4096] {
        let frame = frame_bytes(locals);
        assert!(frame >= locals, "frame {frame} must hold {locals} bytes");
        assert_eq!((4 + 4 + frame) % 16, 0, "frame {frame} for {locals} locals");
    }
}

#[test]
fn frame_slots_descend_from_the_frame_pointer_in_eight_byte_steps() {
    assert_eq!(slot_offset(0), -8);
    assert_eq!(slot_offset(1), -16);
    assert_eq!(slot_offset(7), -64);
}

#[test]
#[allow(clippy::expect_used)]
fn debug_lines_follow_source_locations_and_skip_synthesized_instructions() {
    use crate::{mir::Module, mir::codegen, mir_codegen::MirCodegenSettings, parser};
    use lamina_platform::TargetOperatingSystem;

    const SOURCE: &str = "fn @main() -> i32 {\n  entry:\n    %a = add.i32 10, 20\n    %b = mul.i32 %a, 3\n    %c = sub.i32 %b, 7\n    jmp tail\n\n  tail:\n    ret.i32 %c\n}\n";

    let ir = parser::parse_module(SOURCE).expect("parse");
    let mut module = codegen::from_ir_x86_64(&ir, "runtime_symbols").expect("lower");
    let settings = MirCodegenSettings {
        emit_asm_debug_lines: true,
        ..Default::default()
    };
    let lines = |module: &Module| -> Vec<String> {
        let mut out = Vec::new();
        super::generate_mir_i686_with_units_and_settings(
            module,
            &mut out,
            TargetOperatingSystem::Linux,
            1,
            &settings,
        )
        .expect("emit");
        String::from_utf8(out)
            .expect("utf8")
            .lines()
            .filter(|line| line.contains(".loc "))
            .map(str::to_string)
            .collect()
    };
    assert_eq!(
        lines(&module),
        [
            "    .loc 1 3 5",
            "    .loc 1 4 5",
            "    .loc 1 5 5",
            "    .loc 1 9 5",
        ]
    );

    for function in module.functions.values_mut() {
        for block in &mut function.blocks {
            for inst in &mut block.instructions {
                if inst.loc.map(|loc| loc.line()) == Some(4) {
                    inst.loc = None;
                }
            }
        }
    }
    assert_eq!(
        lines(&module),
        ["    .loc 1 3 5", "    .loc 1 5 5", "    .loc 1 9 5"]
    );
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod aggregate_return_tests {
    use crate::mir::codegen::from_ir_for_target;
    use crate::parser::parse_module;
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    fn emit(source: &str) -> Result<String, String> {
        let ir = parse_module(source).expect("parse");
        let mir = from_ir_for_target(&ir, "i686_test", TargetArchitecture::I686.data_layout())
            .map_err(|error| error.to_string())?;
        let mut out = Vec::new();
        super::super::generate_mir_i686_with_units_and_settings(
            &mir,
            &mut out,
            TargetOperatingSystem::Linux,
            1,
            &Default::default(),
        )
        .map_err(|error| error.to_string())?;
        Ok(String::from_utf8(out).expect("utf8"))
    }

    const ONE_FIELD_LONG: &str = "fn @make(i32 %x) -> struct { a: i64 } {\nentry:\n  \
         %p = stackalloc.struct { a: i64 }\n  %lo = getelem.i32 %p, 0\n  \
         store.i32 %lo, %x\n  ret.struct { a: i64 } %p\n}\n";

    #[test]
    fn a_one_field_sixty_four_bit_struct_comes_back_through_the_hidden_pointer() {
        let assembly = emit(ONE_FIELD_LONG).expect("a struct return is a cdecl memory return");
        assert!(assembly.contains("ret $4"), "{assembly}");
        assert!(assembly.contains("movl 8(%ebp)"), "{assembly}");
        assert!(assembly.contains("movl 12(%ebp)"), "{assembly}");
    }

    #[test]
    fn a_two_word_struct_comes_back_through_the_hidden_pointer() {
        let assembly = emit(
            "fn @make(i32 %x) -> struct { a: i32, b: i32 } {\nentry:\n  \
             %p = stackalloc.struct { a: i32, b: i32 }\n  \
             %pa = getfieldptr.struct { a: i32, b: i32 } %p, 0\n  \
             store.i32 %pa, %x\n  ret.struct { a: i32, b: i32 } %p\n}\n",
        )
        .expect("a struct return is a cdecl memory return");
        assert!(assembly.contains("ret $4"), "{assembly}");
    }

    #[test]
    fn a_legalized_sixty_four_bit_integer_comes_back_in_edx_eax() {
        let assembly = emit("fn @make(i64 %x) -> i64 {\nentry:\n  ret.i64 %x\n}\n")
            .expect("a 64-bit integer returns in %edx:%eax");
        assert!(assembly.contains("movl 4(%ecx), %edx"), "{assembly}");
        assert!(assembly.contains("movl (%ecx), %eax"), "{assembly}");
        assert!(!assembly.contains("ret $4"), "{assembly}");
    }

    #[test]
    fn a_call_reads_a_legalized_sixty_four_bit_result_out_of_edx_eax() {
        let assembly = emit(
            "@extern\nfn @make(i64 %x) -> i64\n\nfn @use(i64 %x) -> i32 {\nentry:\n  \
             %v = call @make(%x)\n  %n = trunc.i64.i32 %v\n  ret.i32 %n\n}\n",
        )
        .expect("a 64-bit result arrives in %edx:%eax");
        assert!(assembly.contains("call make"), "{assembly}");
        assert!(assembly.contains("movl %eax, -"), "{assembly}");
        assert!(assembly.contains("movl %edx, -"), "{assembly}");
    }
}

#[test]
fn a_narrow_unsigned_integer_reaches_a_float_through_the_doubleword_fild() {
    use super::generate_mir_i686_with_units_and_settings;
    use crate::mir::codegen::from_ir_for_target;
    use crate::mir_codegen::MirCodegenSettings;
    use crate::parser::parse_module;
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    for (from, to, unsigned) in [
        ("u8", "f32", true),
        ("u16", "f64", true),
        ("u32", "f32", true),
        ("u32", "f64", true),
        ("i8", "f32", false),
        ("i16", "f64", false),
        ("i32", "f32", false),
    ] {
        let source = format!(
            "fn @conv({from} %x, ptr %o) -> void {{\nentry:\n  %r = inttofp.{from}.{to} %x\n  store.{to} %o, %r\n  ret.void\n}}\n"
        );
        let module = parse_module(&source).expect("parse");
        let mir = from_ir_for_target(
            &module,
            "i686_inttofp",
            TargetArchitecture::I686.data_layout(),
        )
        .expect("lower");
        let mut output = Vec::new();
        generate_mir_i686_with_units_and_settings(
            &mir,
            &mut output,
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings::default(),
        )
        .expect("i686 should lower a narrow integer to float");
        let assembly = String::from_utf8(output).expect("utf8");
        if unsigned {
            assert!(
                assembly.contains("    pushl $0") && assembly.contains("    fildll (%esp)"),
                "{from} -> {to} did not zero-extend into a doubleword fild:\n{assembly}"
            );
        } else {
            assert!(
                assembly.contains("    fildl (%esp)") && !assembly.contains("    fildll (%esp)"),
                "{from} -> {to} did not use the word fild:\n{assembly}"
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod ordered_comparison_tests {
    use crate::mir::codegen::from_ir_for_target;
    use crate::mir_codegen::MirCodegenSettings;
    use crate::parser::parse_module;
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    fn emit(source: &str) -> String {
        let ir = parse_module(source).expect("parse");
        let mir = from_ir_for_target(&ir, "i686_test", TargetArchitecture::I686.data_layout())
            .expect("lower");
        let mut out = Vec::new();
        super::super::generate_mir_i686_with_units_and_settings(
            &mir,
            &mut out,
            TargetOperatingSystem::Linux,
            1,
            &MirCodegenSettings::default(),
        )
        .expect("emit");
        String::from_utf8(out).expect("utf8")
    }

    fn compare(op: &str) -> String {
        emit(&format!(
            "fn @c(f64 %a, f64 %b) -> bool {{\nentry:\n  \
             %r = {op}.f64 %a, %b\n  ret.bool %r\n}}\n"
        ))
    }

    /// An unordered compare sets CF and ZF alike, so `setb` and `setbe` answer true
    /// for NaN. An ordered `lt` and `le` must not, and are taken as the reverse
    /// `gt` and `ge` instead.
    #[test]
    fn an_unordered_operand_makes_every_ordered_comparison_false() {
        for op in ["lt", "le", "gt", "ge"] {
            let assembly = compare(op);
            assert!(
                !assembly.contains("    setb ") && !assembly.contains("    setbe "),
                "{op} answers true for an unordered compare:\n{assembly}"
            );
        }
    }

    #[test]
    fn a_reversed_comparison_puts_its_operands_on_the_stack_the_other_way_round() {
        let loads = |assembly: &str| {
            assembly
                .lines()
                .filter_map(|line| line.trim().strip_prefix("fldl "))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        let less = compare("lt");
        let greater = compare("gt");
        let (mut reversed, forward) = (loads(&less), loads(&greater));
        assert_eq!(reversed.len(), 2, "{less}");
        reversed.reverse();
        assert_eq!(
            reversed, forward,
            "lt did not load the x87 stack the other way round from gt:\n{less}\n{greater}"
        );
    }

    /// A boolean is a byte everywhere else in the compiler, so `true + true` is two,
    /// not the one bit an `andl $1` would leave.
    #[test]
    fn boolean_arithmetic_wraps_at_a_byte() {
        let assembly = emit(
            "fn @c(bool %a, bool %b) -> bool {\nentry:\n  \
             %r = add.bool %a, %b\n  ret.bool %r\n}\n",
        );
        assert!(
            !assembly.contains("    andl $1,"),
            "boolean arithmetic was masked to one bit:\n{assembly}"
        );
        assert!(
            assembly.contains("    movzbl %al, %eax"),
            "boolean arithmetic did not wrap at a byte:\n{assembly}"
        );
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod variadic_wide_argument_tests {
    use crate::mir::codegen::from_ir_for_target;
    use crate::parser::parse_module;
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    fn emit(source: &str) -> String {
        let ir = parse_module(source).expect("parse");
        let mir = from_ir_for_target(&ir, "i686_variadic", TargetArchitecture::I686.data_layout())
            .expect("lower");
        let mut out = Vec::new();
        super::super::generate_mir_i686_with_units_and_settings(
            &mir,
            &mut out,
            TargetOperatingSystem::Linux,
            1,
            &Default::default(),
        )
        .expect("codegen");
        String::from_utf8(out).expect("utf8")
    }

    #[test]
    fn a_legalized_pair_in_the_variadic_tail_travels_as_its_two_words() {
        let assembly = emit(
            "@extern fn @printf(ptr %fmt, ...) -> i32\n\
             global $fmt: [6 x i8] = \"%lld\\n\"\n\
             fn @main() -> i32 {\nentry:\n  %x = add.i64 1, 2\n  \
             %r = call @printf(ptr $fmt, i64 %x)\n  ret.i32 %r\n}\n",
        );
        assert!(assembly.contains("    movl 0(%ecx), %eax"), "{assembly}");
        assert!(assembly.contains("    movl %eax, 4(%esp)"), "{assembly}");
        assert!(assembly.contains("    movl 4(%ecx), %eax"), "{assembly}");
        assert!(assembly.contains("    movl %eax, 8(%esp)"), "{assembly}");
    }

    #[test]
    fn a_named_wide_parameter_of_a_variadic_callee_travels_as_its_two_words() {
        let assembly = emit(
            "@extern fn @take(i64 %a, ...) -> i32\n\
             fn @main() -> i32 {\nentry:\n  %x = add.i64 1, 2\n  \
             %r = call @take(i64 %x, i32 5)\n  ret.i32 %r\n}\n",
        );
        assert!(assembly.contains("    movl 0(%ecx), %eax"), "{assembly}");
        assert!(assembly.contains("    movl %eax, 0(%esp)"), "{assembly}");
        assert!(assembly.contains("    movl 4(%ecx), %eax"), "{assembly}");
        assert!(assembly.contains("    movl %eax, 4(%esp)"), "{assembly}");
        assert!(assembly.contains("    movl %eax, 8(%esp)"), "{assembly}");
    }

    const WIDE_VA_ARG: &str = "fn @read(i32 %n, ...) -> i64 {\nentry:\n  %ap = stackalloc.valist\n  va_start %ap\n  %x = va_arg.i64 %ap\n  va_end %ap\n  ret.i64 %x\n}\n";

    #[test]
    fn va_start_points_past_the_named_parameters() {
        let assembly = emit(WIDE_VA_ARG);
        assert!(assembly.contains("    leal 12(%ebp), %eax"), "{assembly}");
    }

    #[test]
    fn va_arg_of_a_pair_reads_two_packed_words_and_steps_by_eight() {
        let assembly = emit(WIDE_VA_ARG);
        assert!(assembly.contains("    movl (%edx), %eax"), "{assembly}");
        assert!(assembly.contains("    movl 4(%edx), %eax"), "{assembly}");
        assert!(assembly.contains("    addl $8, %edx"), "{assembly}");
    }

    #[test]
    fn va_arg_of_a_word_steps_by_four() {
        let assembly = emit(
            "fn @read(i32 %n, ...) -> i32 {\nentry:\n  %ap = stackalloc.valist\n  \
             va_start %ap\n  %x = va_arg.i32 %ap\n  va_end %ap\n  ret.i32 %x\n}\n",
        );
        assert!(assembly.contains("    addl $4, %edx"), "{assembly}");
        assert!(!assembly.contains("    addl $8, %edx"), "{assembly}");
    }
}
