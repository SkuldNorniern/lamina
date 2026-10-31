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
