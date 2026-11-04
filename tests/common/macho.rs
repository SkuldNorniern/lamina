#![allow(dead_code)]

use std::str;

pub const LC_SEGMENT_64: u32 = 0x19;
pub const LC_SYMTAB: u32 = 0x2;
pub const N_EXT: u8 = 0x1;
pub const N_TYPE: u8 = 0x0e;
pub const N_SECT: u8 = 0x0e;
pub const MH_HAS_TLV_DESCRIPTORS: u32 = 0x0080_0000;

pub struct Section {
    pub segment: String,
    pub name: String,
    pub addr: u64,
    pub size: u64,
    pub offset: u32,
    pub align: u32,
    pub flags: u32,
    pub data: Vec<u8>,
    pub relocs: Vec<Reloc>,
}

#[derive(Clone, Copy, Debug)]
pub struct Reloc {
    pub address: u32,
    pub symbolnum: u32,
    pub pcrel: bool,
    pub length: u32,
    pub external: bool,
    pub kind: u32,
}

pub struct Symbol {
    pub name: String,
    pub n_type: u8,
    pub n_sect: u8,
    pub n_desc: u16,
    pub n_value: u64,
}

impl Symbol {
    pub fn external(&self) -> bool {
        self.n_type & N_EXT != 0
    }

    pub fn defined_in_section(&self) -> bool {
        self.n_type & N_TYPE == N_SECT
    }
}

pub struct MachO {
    pub flags: u32,
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

fn name16(bytes: &[u8], at: usize) -> String {
    let field = &bytes[at..at + 16];
    let end = field.iter().position(|&b| b == 0).unwrap_or(16);
    str::from_utf8(&field[..end])
        .expect("utf-8 section name")
        .to_string()
}

impl MachO {
    pub fn parse(object: &[u8]) -> Self {
        assert_eq!(u32_at(object, 0), 0xfeed_facf, "Mach-O 64 magic");
        let ncmds = u32_at(object, 16);
        let flags = u32_at(object, 24);
        let mut sections = Vec::new();
        let mut symbols = Vec::new();
        let mut at = 32;
        for _ in 0..ncmds {
            let cmd = u32_at(object, at);
            let size = u32_at(object, at + 4) as usize;
            if cmd == LC_SEGMENT_64 {
                let nsects = u32_at(object, at + 64) as usize;
                for index in 0..nsects {
                    let head = at + 72 + index * 80;
                    let offset = u32_at(object, head + 48);
                    let size = u64_at(object, head + 40);
                    let reloff = u32_at(object, head + 56) as usize;
                    let nreloc = u32_at(object, head + 60) as usize;
                    let flags = u32_at(object, head + 64);
                    let zerofill = matches!(flags & 0xff, 0x1 | 0x12);
                    let data = if zerofill {
                        vec![0; size as usize]
                    } else {
                        object[offset as usize..offset as usize + size as usize].to_vec()
                    };
                    let relocs = (0..nreloc)
                        .map(|i| {
                            let word = u32_at(object, reloff + i * 8 + 4);
                            Reloc {
                                address: u32_at(object, reloff + i * 8),
                                symbolnum: word & 0x00ff_ffff,
                                pcrel: word >> 24 & 1 != 0,
                                length: word >> 25 & 3,
                                external: word >> 27 & 1 != 0,
                                kind: word >> 28 & 0xf,
                            }
                        })
                        .collect();
                    sections.push(Section {
                        name: name16(object, head),
                        segment: name16(object, head + 16),
                        addr: u64_at(object, head + 32),
                        size,
                        offset,
                        align: u32_at(object, head + 52),
                        flags,
                        data,
                        relocs,
                    });
                }
            } else if cmd == LC_SYMTAB {
                let symoff = u32_at(object, at + 8) as usize;
                let nsyms = u32_at(object, at + 12) as usize;
                let stroff = u32_at(object, at + 16) as usize;
                for i in 0..nsyms {
                    let entry = symoff + i * 16;
                    let strx = u32_at(object, entry) as usize;
                    let start = stroff + strx;
                    let end = start
                        + object[start..]
                            .iter()
                            .position(|&b| b == 0)
                            .expect("nul terminated");
                    symbols.push(Symbol {
                        name: str::from_utf8(&object[start..end])
                            .expect("utf-8 symbol")
                            .to_string(),
                        n_type: object[entry + 4],
                        n_sect: object[entry + 5],
                        n_desc: u16::from_le_bytes([object[entry + 6], object[entry + 7]]),
                        n_value: u64_at(object, entry + 8),
                    });
                }
            }
            at += size;
        }
        Self {
            flags,
            sections,
            symbols,
        }
    }

    pub fn section(&self, segment: &str, name: &str) -> &Section {
        self.sections
            .iter()
            .find(|s| s.segment == segment && s.name == name)
            .unwrap_or_else(|| panic!("object has no {segment},{name}"))
    }

    pub fn find_section(&self, segment: &str, name: &str) -> Option<&Section> {
        self.sections
            .iter()
            .find(|s| s.segment == segment && s.name == name)
    }

    pub fn symbol(&self, name: &str) -> &Symbol {
        self.symbols
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("object has no symbol {name}"))
    }

    /// The one-based section index a `n_sect` field names, as a `segment,section` pair.
    pub fn section_of(&self, symbol: &Symbol) -> &Section {
        &self.sections[symbol.n_sect as usize - 1]
    }

    pub fn symbol_name(&self, index: u32) -> &str {
        &self.symbols[index as usize].name
    }
}
