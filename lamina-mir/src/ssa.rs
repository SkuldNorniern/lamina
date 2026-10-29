//! Read-only SSA value graph over an immutable LUMIR snapshot. Merges are block
//! parameters. Block 0 is a synthetic entry that carries the function parameters, so
//! a backedge into the real entry never stands in for the first invocation.

use crate::loc::Located;
use lamina_platform::PointerWidth;
use std::collections::HashMap;
use std::fmt;

use crate::{
    block::Block,
    function::Function,
    instruction::Instruction,
    io_intrinsic_result_type,
    module::Module,
    register::Register,
    types::MirType,
    verify::{Dominators, dominator_tree, instruction_definition_type},
};

mod destroy;

pub use destroy::{DestroyError, destroy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

pub const SYNTHETIC_ENTRY: BlockId = BlockId(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueDef {
    BlockParam { block: BlockId, index: u32 },
    Inst { block: BlockId, inst: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SsaValue {
    pub def: ValueDef,
    pub ty: Option<MirType>,
    pub register: Register,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SsaInst {
    pub source: usize,
    pub defines: Option<ValueId>,
    pub uses: Vec<ValueId>,
    pub instruction: Located,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SsaEdge {
    pub target: BlockId,
    pub args: Vec<ValueId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SsaBlock {
    pub label: String,
    pub source: Option<usize>,
    /// Emission rank, ordered lexicographically. An original block at index `b` ranks
    /// `(b, 0)`; a block a pass inserts before it takes the next rank below.
    pub order: (usize, i64),
    pub params: Vec<ValueId>,
    pub insts: Vec<SsaInst>,
    pub edges: Vec<SsaEdge>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SsaFunction {
    pub name: String,
    pub values: Vec<SsaValue>,
    pub blocks: Vec<SsaBlock>,
    pub unreachable: Vec<(usize, Block)>,
    pub shell: Function,
    pub dominators: Dominators,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaBuildError {
    MissingEntry {
        function: String,
        entry: String,
    },
    UnknownTarget {
        function: String,
        block: String,
        target: String,
    },
    PhysicalRegister {
        function: String,
        register: String,
    },
    DuplicateParameter {
        function: String,
        register: String,
    },
    Unassigned {
        function: String,
        register: String,
        block: String,
    },
    ConflictingTypes {
        function: String,
        register: String,
        first: MirType,
        second: MirType,
    },
}

impl fmt::Display for SsaBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaBuildError::MissingEntry { function, entry } => {
                write!(f, "entry block '{entry}' of '@{function}' does not exist")
            }
            SsaBuildError::UnknownTarget {
                function,
                block,
                target,
            } => write!(
                f,
                "block '{block}' of '@{function}' branches to unknown block '{target}'"
            ),
            SsaBuildError::PhysicalRegister { function, register } => write!(
                f,
                "physical register '{register}' in '@{function}' is outside the SSA graph"
            ),
            SsaBuildError::DuplicateParameter { function, register } => {
                write!(f, "parameter '{register}' of '@{function}' is listed twice")
            }
            SsaBuildError::Unassigned {
                function,
                register,
                block,
            } => write!(
                f,
                "'{register}' is read in '@{function}' on a path from the entry with no assignment (block '{block}')"
            ),
            SsaBuildError::ConflictingTypes {
                function,
                register,
                first,
                second,
            } => write!(
                f,
                "'{register}' in '@{function}' is assigned both {first} and {second}"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsaRule {
    /// Every use names a value, and every value has exactly one definition site.
    SingleDefinition,
    /// Every use is dominated by its definition; within a block, a definition precedes its use.
    Dominance,
    /// Every edge passes one argument per parameter of its target.
    EdgeArity,
    /// Every edge argument has the type of the parameter it feeds, where both are known.
    EdgeTypes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaViolation {
    pub rule: SsaRule,
    pub block: usize,
    pub message: String,
}

fn result_type(
    instruction: &Instruction,
    callee_return: &dyn Fn(&str) -> Option<MirType>,
    pointer: PointerWidth,
) -> Option<MirType> {
    match instruction {
        Instruction::Call { name, .. } => {
            callee_return(name).or_else(|| io_intrinsic_result_type(name, pointer))
        }
        Instruction::CallIndirect { ret_ty, .. } => *ret_ty,
        Instruction::AtomicLoad { ty, .. }
        | Instruction::AtomicBinary { ty, .. }
        | Instruction::AtomicCompareExchange { ty, .. }
        | Instruction::VaArg { ty, .. } => Some(*ty),
        #[cfg(feature = "nightly")]
        Instruction::SimdBinary { ty, .. }
        | Instruction::SimdUnary { ty, .. }
        | Instruction::SimdTernary { ty, .. }
        | Instruction::SimdShuffle { ty, .. }
        | Instruction::SimdInsert { ty, .. }
        | Instruction::SimdLoad { ty, .. } => Some(*ty),
        _ => instruction_definition_type(instruction),
    }
}

struct Registers<'f> {
    function: &'f str,
    index: HashMap<Register, usize>,
    list: Vec<Register>,
}

impl Registers<'_> {
    fn intern(&mut self, register: &Register) -> Result<usize, SsaBuildError> {
        if let Register::Physical(_) = register {
            return Err(SsaBuildError::PhysicalRegister {
                function: self.function.to_string(),
                register: register.to_string(),
            });
        }
        if let Some(&i) = self.index.get(register) {
            return Ok(i);
        }
        let i = self.list.len();
        self.index.insert(register.clone(), i);
        self.list.push(register.clone());
        Ok(i)
    }

    fn get(&self, register: &Register) -> usize {
        self.index[register]
    }
}

type Bits = Vec<u64>;

fn bit_set(bits: &mut [u64], i: usize) {
    bits[i / 64] |= 1 << (i % 64);
}

fn bit_get(bits: &[u64], i: usize) -> bool {
    bits[i / 64] >> (i % 64) & 1 == 1
}

fn bits_iter(bits: &[u64]) -> impl Iterator<Item = usize> + '_ {
    bits.iter().enumerate().flat_map(|(word, &bits)| {
        (0..64usize)
            .filter(move |bit| bits >> bit & 1 == 1)
            .map(move |bit| word * 64 + bit)
    })
}

impl SsaFunction {
    pub fn build(
        function: &Function,
        callee_return: &dyn Fn(&str) -> Option<MirType>,
    ) -> Result<Self, SsaBuildError> {
        let mut graph = Self::construct(function, callee_return)?;
        graph.prune_trivial_parameters();
        Ok(graph)
    }

    fn construct(
        function: &Function,
        callee_return: &dyn Fn(&str) -> Option<MirType>,
    ) -> Result<Self, SsaBuildError> {
        let name = function.sig.name.as_str();
        let block_count = function.blocks.len();
        let index_of: HashMap<&str, usize> = function
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| (block.label.as_str(), i))
            .collect();
        let entry =
            *index_of
                .get(function.entry.as_str())
                .ok_or_else(|| SsaBuildError::MissingEntry {
                    function: name.to_string(),
                    entry: function.entry.clone(),
                })?;

        let mut succ: Vec<Vec<usize>> = Vec::with_capacity(block_count);
        for block in &function.blocks {
            let mut targets = Vec::new();
            for label in block.successors() {
                let &target = index_of
                    .get(label)
                    .ok_or_else(|| SsaBuildError::UnknownTarget {
                        function: name.to_string(),
                        block: block.label.clone(),
                        target: label.to_string(),
                    })?;
                targets.push(target);
            }
            succ.push(targets);
        }

        let mut reachable = vec![false; block_count];
        reachable[entry] = true;
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            for &s in &succ[b] {
                if !reachable[s] {
                    reachable[s] = true;
                    stack.push(s);
                }
            }
        }
        let order: Vec<usize> = (0..block_count).filter(|&b| reachable[b]).collect();
        let unreachable = (0..block_count)
            .filter(|&b| !reachable[b])
            .map(|b| (b, function.blocks[b].clone()))
            .collect();

        let mut regs = Registers {
            function: name,
            index: HashMap::new(),
            list: Vec::new(),
        };
        let mut param_regs = Vec::with_capacity(function.sig.params.len());
        for param in &function.sig.params {
            let r = regs.intern(&param.reg)?;
            if param_regs.contains(&r) {
                return Err(SsaBuildError::DuplicateParameter {
                    function: name.to_string(),
                    register: param.reg.to_string(),
                });
            }
            param_regs.push(r);
        }
        for &b in &order {
            for instr in &function.blocks[b].instructions {
                for reg in instr.use_regs() {
                    regs.intern(reg)?;
                }
                if let Some(reg) = instr.def_reg() {
                    regs.intern(reg)?;
                }
            }
        }
        let nregs = regs.list.len();
        let words = nregs.div_ceil(64);

        let mut ue = vec![vec![0u64; words]; block_count];
        let mut defs = vec![vec![0u64; words]; block_count];
        for &b in &order {
            for instr in &function.blocks[b].instructions {
                for reg in instr.use_regs() {
                    let r = regs.get(reg);
                    if !bit_get(&defs[b], r) {
                        bit_set(&mut ue[b], r);
                    }
                }
                if let Some(reg) = instr.def_reg() {
                    bit_set(&mut defs[b], regs.get(reg));
                }
            }
        }

        let mut live_in: Vec<Bits> = vec![vec![0u64; words]; block_count];
        let mut live_out: Vec<Bits> = vec![vec![0u64; words]; block_count];
        loop {
            let mut changed = false;
            for &b in order.iter().rev() {
                let mut out = vec![0u64; words];
                for &s in &succ[b] {
                    for (o, i) in out.iter_mut().zip(&live_in[s]) {
                        *o |= i;
                    }
                }
                let inn: Bits = (0..words)
                    .map(|w| ue[b][w] | (out[w] & !defs[b][w]))
                    .collect();
                if out != live_out[b] || inn != live_in[b] {
                    changed = true;
                    live_out[b] = out;
                    live_in[b] = inn;
                }
            }
            if !changed {
                break;
            }
        }

        let mut is_param = vec![0u64; words];
        for &r in &param_regs {
            bit_set(&mut is_param, r);
        }
        if let Some(r) = bits_iter(&live_in[entry]).find(|&r| !bit_get(&is_param, r)) {
            return Err(SsaBuildError::Unassigned {
                function: name.to_string(),
                register: regs.list[r].to_string(),
                block: function.blocks[entry].label.clone(),
            });
        }

        let mut reg_type: Vec<Option<MirType>> = vec![None; nregs];
        let mut unify = |r: usize, ty: MirType| -> Result<(), SsaBuildError> {
            match reg_type[r] {
                None => {
                    reg_type[r] = Some(ty);
                    Ok(())
                }
                Some(first) if first == ty => Ok(()),
                Some(first) => Err(SsaBuildError::ConflictingTypes {
                    function: name.to_string(),
                    register: regs.list[r].to_string(),
                    first,
                    second: ty,
                }),
            }
        };
        for (param, &r) in function.sig.params.iter().zip(&param_regs) {
            unify(r, param.ty)?;
        }
        let mut inst_types: Vec<Vec<Option<MirType>>> = vec![Vec::new(); block_count];
        for &b in &order {
            for instr in &function.blocks[b].instructions {
                let ty = result_type(instr, callee_return, function.pointer_width);
                if let (Some(reg), Some(ty)) = (instr.def_reg(), ty) {
                    unify(regs.get(reg), ty)?;
                }
                inst_types[b].push(ty);
            }
        }

        let mut values: Vec<SsaValue> = Vec::new();
        let mut pre_params = Vec::with_capacity(param_regs.len());
        for (index, (param, &r)) in function.sig.params.iter().zip(&param_regs).enumerate() {
            pre_params.push(ValueId(values.len() as u32));
            values.push(SsaValue {
                def: ValueDef::BlockParam {
                    block: SYNTHETIC_ENTRY,
                    index: index as u32,
                },
                ty: Some(param.ty),
                register: regs.list[r].clone(),
            });
        }

        let mut ssa_of: Vec<Option<usize>> = vec![None; block_count];
        let mut block_param_regs: Vec<Vec<usize>> = vec![Vec::new(); block_count];
        let mut block_params: Vec<Vec<ValueId>> = vec![Vec::new(); block_count];
        let mut result_of: Vec<Vec<Option<ValueId>>> = vec![Vec::new(); block_count];
        for (k, &b) in order.iter().enumerate() {
            ssa_of[b] = Some(k + 1);
            block_param_regs[b] = bits_iter(&live_in[b]).collect();
            for (index, &r) in block_param_regs[b].iter().enumerate() {
                block_params[b].push(ValueId(values.len() as u32));
                values.push(SsaValue {
                    def: ValueDef::BlockParam {
                        block: BlockId((k + 1) as u32),
                        index: index as u32,
                    },
                    ty: reg_type[r],
                    register: regs.list[r].clone(),
                });
            }
            for (i, instr) in function.blocks[b].instructions.iter().enumerate() {
                let defines = instr.def_reg().map(|reg| {
                    let id = ValueId(values.len() as u32);
                    values.push(SsaValue {
                        def: ValueDef::Inst {
                            block: BlockId((k + 1) as u32),
                            inst: i as u32,
                        },
                        ty: inst_types[b][i].or(reg_type[regs.get(reg)]),
                        register: reg.clone(),
                    });
                    id
                });
                result_of[b].push(defines);
            }
        }

        let unassigned = |r: usize, block: usize| SsaBuildError::Unassigned {
            function: name.to_string(),
            register: regs.list[r].to_string(),
            block: function.blocks[block].label.clone(),
        };

        let mut blocks = Vec::with_capacity(order.len() + 1);
        let mut entry_args = Vec::with_capacity(block_param_regs[entry].len());
        for &r in &block_param_regs[entry] {
            let position = param_regs
                .iter()
                .position(|&p| p == r)
                .ok_or_else(|| unassigned(r, entry))?;
            entry_args.push(pre_params[position]);
        }
        blocks.push(SsaBlock {
            label: "<entry>".to_string(),
            source: None,
            order: (0, 0),
            params: pre_params,
            insts: Vec::new(),
            edges: vec![SsaEdge {
                target: BlockId(ssa_of[entry].unwrap_or(0) as u32),
                args: entry_args,
            }],
        });

        for &b in &order {
            let block = &function.blocks[b];
            let mut current: HashMap<usize, ValueId> = block_param_regs[b]
                .iter()
                .copied()
                .zip(block_params[b].iter().copied())
                .collect();
            let mut insts = Vec::with_capacity(block.instructions.len());
            for (i, instr) in block.instructions.iter().enumerate() {
                let mut uses = Vec::new();
                for reg in instr.use_regs() {
                    let r = regs.get(reg);
                    let value = *current.get(&r).ok_or_else(|| unassigned(r, b))?;
                    uses.push(value);
                }
                let defines = result_of[b][i];
                if let (Some(reg), Some(id)) = (instr.def_reg(), defines) {
                    current.insert(regs.get(reg), id);
                }
                insts.push(SsaInst {
                    source: i,
                    defines,
                    uses,
                    instruction: instr.clone(),
                });
            }
            let mut edges = Vec::new();
            for label in block.successors() {
                let target = index_of[label];
                let mut args = Vec::with_capacity(block_param_regs[target].len());
                for &r in &block_param_regs[target] {
                    args.push(*current.get(&r).ok_or_else(|| unassigned(r, b))?);
                }
                edges.push(SsaEdge {
                    target: BlockId(ssa_of[target].unwrap_or(0) as u32),
                    args,
                });
            }
            blocks.push(SsaBlock {
                label: block.label.clone(),
                source: Some(b),
                order: (b, 0),
                params: block_params[b].clone(),
                insts,
                edges,
            });
        }

        let successors: Vec<Vec<usize>> = blocks
            .iter()
            .map(|block| {
                block
                    .edges
                    .iter()
                    .map(|edge| edge.target.0 as usize)
                    .collect()
            })
            .collect();
        let dominators = dominator_tree(&successors, (!blocks.is_empty()).then_some(0));
        let mut shell = function.clone();
        shell.blocks.clear();

        Ok(SsaFunction {
            name: name.to_string(),
            values,
            blocks,
            unreachable,
            shell,
            dominators,
        })
    }

    /// Recomputes the dominator tree after a pass changed the edges or added blocks.
    pub fn recompute_dominators(&mut self) {
        let successors: Vec<Vec<usize>> = self
            .blocks
            .iter()
            .map(|block| {
                block
                    .edges
                    .iter()
                    .map(|edge| edge.target.0 as usize)
                    .collect()
            })
            .collect();
        self.dominators = dominator_tree(&successors, (!self.blocks.is_empty()).then_some(0));
    }

    /// Records a value defined by `def` that carries `register` of type `ty`.
    pub fn add_value(&mut self, def: ValueDef, ty: Option<MirType>, register: Register) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(SsaValue { def, ty, register });
        id
    }

    /// A label no block in the graph holds, counting the unreachable blocks that
    /// destruction places back beside them.
    pub fn fresh_label(&self, base: &str) -> String {
        let taken = |label: &str| {
            self.blocks.iter().any(|block| block.label == label)
                || self
                    .unreachable
                    .iter()
                    .any(|(_, block)| block.label == label)
        };
        let mut label = base.to_string();
        while taken(&label) {
            label.push('_');
        }
        label
    }

    fn prune_trivial_parameters(&mut self) {
        let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.blocks.len()];
        for (b, block) in self.blocks.iter().enumerate() {
            for (e, edge) in block.edges.iter().enumerate() {
                incoming[edge.target.0 as usize].push((b, e));
            }
        }

        let mut replaced: Vec<Option<ValueId>> = vec![None; self.values.len()];
        let resolve = |replaced: &[Option<ValueId>], mut v: ValueId| {
            while let Some(next) = replaced[v.0 as usize] {
                v = next;
            }
            v
        };

        let mut changed = true;
        while changed {
            changed = false;
            for (b, incoming_b) in incoming.iter().enumerate().skip(1) {
                let mut i = 0;
                while i < self.blocks[b].params.len() {
                    let param = self.blocks[b].params[i];
                    let mut same: Option<ValueId> = None;
                    let mut trivial = true;
                    for &(pred, e) in incoming_b {
                        let arg = resolve(&replaced, self.blocks[pred].edges[e].args[i]);
                        if arg == param {
                            continue;
                        }
                        match same {
                            None => same = Some(arg),
                            Some(first) if first == arg => {}
                            Some(_) => trivial = false,
                        }
                    }
                    match same {
                        Some(value) if trivial => {
                            replaced[param.0 as usize] = Some(value);
                            self.blocks[b].params.remove(i);
                            for &(pred, e) in incoming_b {
                                self.blocks[pred].edges[e].args.remove(i);
                            }
                            changed = true;
                        }
                        _ => i += 1,
                    }
                }
            }
        }

        if replaced.iter().all(Option::is_none) {
            return;
        }
        let resolved: Vec<Option<ValueId>> = (0..self.values.len())
            .map(|v| replaced[v].map(|_| resolve(&replaced, ValueId(v as u32))))
            .collect();
        let mut renumber: Vec<Option<u32>> = vec![None; self.values.len()];
        let mut kept = Vec::with_capacity(self.values.len());
        for (v, value) in self.values.drain(..).enumerate() {
            if resolved[v].is_none() {
                renumber[v] = Some(kept.len() as u32);
                kept.push(value);
            }
        }
        let map = |id: &mut ValueId| {
            let target = resolved[id.0 as usize].unwrap_or(*id);
            *id = ValueId(renumber[target.0 as usize].unwrap_or(0));
        };
        for block in &mut self.blocks {
            for id in &mut block.params {
                map(id);
            }
            for inst in &mut block.insts {
                for id in &mut inst.uses {
                    map(id);
                }
                if let Some(id) = &mut inst.defines {
                    map(id);
                }
            }
            for edge in &mut block.edges {
                for id in &mut edge.args {
                    map(id);
                }
            }
        }
        for (b, block) in self.blocks.iter().enumerate() {
            for (index, &p) in block.params.iter().enumerate() {
                kept[p.0 as usize].def = ValueDef::BlockParam {
                    block: BlockId(b as u32),
                    index: index as u32,
                };
            }
        }
        self.values = kept;
    }

    pub fn verify(&self) -> Result<(), Vec<SsaViolation>> {
        let mut out = Vec::new();
        let mut definitions = vec![0usize; self.values.len()];

        let mut note =
            |out: &mut Vec<SsaViolation>, block: usize, id: ValueId, expected: ValueDef| {
                match definitions.get_mut(id.0 as usize) {
                    None => out.push(SsaViolation {
                        rule: SsaRule::SingleDefinition,
                        block,
                        message: format!("definition names missing value v{}", id.0),
                    }),
                    Some(count) => {
                        *count += 1;
                        let recorded = self.values[id.0 as usize].def;
                        if recorded != expected {
                            out.push(SsaViolation {
                                rule: SsaRule::SingleDefinition,
                                block,
                                message: format!(
                                    "v{} is recorded as {recorded:?} but defined at {expected:?}",
                                    id.0
                                ),
                            });
                        }
                    }
                }
            };
        for (b, block) in self.blocks.iter().enumerate() {
            for (index, &p) in block.params.iter().enumerate() {
                note(
                    &mut out,
                    b,
                    p,
                    ValueDef::BlockParam {
                        block: BlockId(b as u32),
                        index: index as u32,
                    },
                );
            }
            for (inst, site) in block.insts.iter().enumerate() {
                if let Some(id) = site.defines {
                    note(
                        &mut out,
                        b,
                        id,
                        ValueDef::Inst {
                            block: BlockId(b as u32),
                            inst: inst as u32,
                        },
                    );
                }
            }
        }
        for (id, &count) in definitions.iter().enumerate() {
            if count != 1 {
                out.push(SsaViolation {
                    rule: SsaRule::SingleDefinition,
                    block: 0,
                    message: format!("v{id} has {count} definitions"),
                });
            }
        }

        let doms = &self.dominators;

        let check_use =
            |out: &mut Vec<SsaViolation>, id: ValueId, b: usize, position: Option<usize>| {
                let Some(value) = self.values.get(id.0 as usize) else {
                    out.push(SsaViolation {
                        rule: SsaRule::SingleDefinition,
                        block: b,
                        message: format!("use of v{} resolves to no value", id.0),
                    });
                    return;
                };
                let ok = match value.def {
                    ValueDef::BlockParam { block, .. } => {
                        let d = block.0 as usize;
                        d == b || doms.dominates(d, b)
                    }
                    ValueDef::Inst { block, inst } => {
                        let d = block.0 as usize;
                        if d == b {
                            position.is_none_or(|i| (inst as usize) < i)
                        } else {
                            doms.dominates(d, b)
                        }
                    }
                };
                if !ok {
                    out.push(SsaViolation {
                        rule: SsaRule::Dominance,
                        block: b,
                        message: format!(
                            "use of v{} in '{}' is not dominated by its definition {:?}",
                            id.0, self.blocks[b].label, value.def
                        ),
                    });
                }
            };

        for (b, block) in self.blocks.iter().enumerate() {
            if !doms.is_reachable(b) {
                continue;
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &u in &inst.uses {
                    check_use(&mut out, u, b, Some(i));
                }
            }
            for edge in &block.edges {
                for &u in &edge.args {
                    check_use(&mut out, u, b, None);
                }
            }
        }

        for (b, block) in self.blocks.iter().enumerate() {
            for edge in &block.edges {
                let Some(target) = self.blocks.get(edge.target.0 as usize) else {
                    out.push(SsaViolation {
                        rule: SsaRule::EdgeArity,
                        block: b,
                        message: format!(
                            "edge from '{}' names missing block {}",
                            block.label, edge.target.0
                        ),
                    });
                    continue;
                };
                if edge.args.len() != target.params.len() {
                    out.push(SsaViolation {
                        rule: SsaRule::EdgeArity,
                        block: b,
                        message: format!(
                            "edge from '{}' to '{}' passes {} arguments for {} parameters",
                            block.label,
                            target.label,
                            edge.args.len(),
                            target.params.len()
                        ),
                    });
                }
                for (&arg, &param) in edge.args.iter().zip(&target.params) {
                    let (Some(arg), Some(param)) = (
                        self.values.get(arg.0 as usize),
                        self.values.get(param.0 as usize),
                    ) else {
                        continue;
                    };
                    if let (Some(a), Some(p)) = (arg.ty, param.ty)
                        && a != p
                    {
                        out.push(SsaViolation {
                            rule: SsaRule::EdgeTypes,
                            block: b,
                            message: format!(
                                "edge from '{}' to '{}' passes {a} for a {p} parameter",
                                block.label, target.label
                            ),
                        });
                    }
                }
            }
        }

        if out.is_empty() { Ok(()) } else { Err(out) }
    }
}

/// Builds, verifies, prunes and destroys every defined function. Every graph is verified
/// both before and after pruning.
pub fn round_trip_module(module: &mut Module) -> Result<(), String> {
    let mut names: Vec<String> = module.functions.keys().cloned().collect();
    names.sort();
    let mut rebuilt = Vec::with_capacity(names.len());
    {
        let callee_return = |name: &str| module.get_function(name).and_then(|f| f.sig.ret_ty);
        for name in &names {
            let function = &module.functions[name];
            if function.blocks.is_empty() {
                continue;
            }
            let check = |graph: &SsaFunction, stage: &str| {
                graph.verify().map_err(|violations| {
                    let lines: Vec<&str> = violations.iter().map(|v| v.message.as_str()).collect();
                    format!("@{name} {stage}: {}", lines.join("; "))
                })
            };
            let mut graph =
                SsaFunction::construct(function, &callee_return).map_err(|e| e.to_string())?;
            check(&graph, "before pruning")?;
            graph.prune_trivial_parameters();
            check(&graph, "after pruning")?;
            let destroyed = destroy(&graph).map_err(|e| e.to_string())?;
            rebuilt.push((name.clone(), destroyed));
        }
    }
    for (name, function) in rebuilt {
        module.functions.insert(name, function);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        function::FunctionBuilder,
        instruction::{IntBinOp, Operand},
        register::VirtualReg,
    };

    fn gpr(id: u32) -> Register {
        VirtualReg::gpr(id).into()
    }

    fn imm(dst: u32, value: i64) -> Instruction {
        Instruction::Copy {
            ty: MirType::i64(),
            dst: gpr(dst),
            src: Operand::imm_i64(value),
        }
    }

    fn mov(dst: u32, src: u32) -> Instruction {
        Instruction::Copy {
            ty: MirType::i64(),
            dst: gpr(dst),
            src: Operand::v_gpr(src),
        }
    }

    fn ret(src: u32) -> Instruction {
        Instruction::Ret {
            value: Some(Operand::v_gpr(src)),
        }
    }

    fn branch(cond: u32, t: &str, f: &str) -> Instruction {
        Instruction::Br {
            cond: gpr(cond),
            true_target: t.to_string(),
            false_target: f.to_string(),
        }
    }

    fn jump(target: &str) -> Instruction {
        Instruction::Jmp {
            target: target.to_string(),
        }
    }

    fn build(function: &Function) -> Result<SsaFunction, SsaBuildError> {
        SsaFunction::build(function, &|_| None)
    }

    fn checked(function: &Function) -> SsaFunction {
        let graph = build(function).expect("snapshot builds");
        graph.verify().expect("graph verifies");
        graph
    }

    fn diamond() -> Function {
        FunctionBuilder::new("diamond")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(branch(0, "a", "b"))
            .block("a")
            .instr(imm(1, 10))
            .instr(jump("join"))
            .block("b")
            .instr(imm(1, 20))
            .instr(jump("join"))
            .block("join")
            .instr(mov(2, 1))
            .instr(ret(2))
            .build()
    }

    #[test]
    fn straight_line_values_are_stable_and_verify() {
        let f = FunctionBuilder::new("line")
            .block("entry")
            .instr(imm(1, 1))
            .instr(mov(2, 1))
            .instr(ret(2))
            .build();
        let graph = checked(&f);
        assert_eq!(graph.values.len(), 2);
        assert_eq!(graph.blocks[1].insts[1].uses, vec![ValueId(0)]);
        assert_eq!(graph.blocks[1].insts[1].defines, Some(ValueId(1)));
        assert!(graph.unreachable.is_empty());
    }

    #[test]
    fn diamond_merge_is_a_block_parameter() {
        let graph = checked(&diamond());
        let join = graph
            .blocks
            .iter()
            .find(|b| b.label == "join")
            .expect("join");
        assert_eq!(join.params.len(), 1);
        let arm = graph.blocks.iter().find(|b| b.label == "a").expect("a");
        assert_eq!(arm.edges[0].args.len(), 1);
        assert_eq!(graph.blocks[0].edges.len(), 1);
    }

    #[test]
    fn loop_carried_value_is_a_header_parameter() {
        let f = FunctionBuilder::new("count")
            .param(gpr(0), MirType::i64())
            .returns(MirType::i64())
            .block("entry")
            .instr(jump("loop"))
            .block("loop")
            .instr(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::i64(),
                dst: gpr(0),
                lhs: Operand::v_gpr(0),
                rhs: Operand::imm_i64(1),
            })
            .instr(branch(0, "loop", "exit"))
            .block("exit")
            .instr(ret(0))
            .build();
        let graph = checked(&f);
        let header = graph
            .blocks
            .iter()
            .find(|b| b.label == "loop")
            .expect("loop");
        assert_eq!(header.params.len(), 1);
        assert_eq!(header.edges[0].args.len(), 1);
    }

    #[test]
    fn entry_backedge_does_not_stand_in_for_the_first_invocation() {
        let f = FunctionBuilder::new("spin")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(mov(1, 0))
            .instr(branch(0, "entry", "exit"))
            .block("exit")
            .instr(ret(1))
            .build();
        let graph = checked(&f);
        assert!(graph.blocks[1].params.is_empty());
        assert_eq!(graph.blocks[0].edges[0].target, BlockId(1));
        assert_eq!(graph.blocks[1].insts[0].uses, graph.blocks[0].params);
    }

    #[test]
    fn read_assigned_on_one_arm_only_is_rejected() {
        let f = FunctionBuilder::new("half")
            .param(gpr(0), MirType::i1())
            .returns(MirType::i64())
            .block("entry")
            .instr(branch(0, "a", "b"))
            .block("a")
            .instr(imm(1, 10))
            .instr(jump("join"))
            .block("b")
            .instr(jump("join"))
            .block("join")
            .instr(ret(1))
            .build();
        match build(&f) {
            Err(SsaBuildError::Unassigned {
                register, block, ..
            }) => {
                assert_eq!(register, "v1");
                assert_eq!(block, "entry");
            }
            other => panic!("expected an unassigned read, got {other:?}"),
        }
    }

    #[test]
    fn read_of_a_non_parameter_at_entry_is_rejected() {
        let f = FunctionBuilder::new("early")
            .returns(MirType::i64())
            .block("entry")
            .instr(mov(1, 0))
            .instr(ret(1))
            .build();
        assert!(matches!(build(&f), Err(SsaBuildError::Unassigned { .. })));
    }

    #[test]
    fn unreachable_blocks_are_listed_and_excluded() {
        let f = FunctionBuilder::new("dead")
            .returns(MirType::i64())
            .block("entry")
            .instr(imm(1, 1))
            .instr(ret(1))
            .block("dead")
            .instr(mov(2, 1))
            .instr(ret(2))
            .build();
        let graph = checked(&f);
        assert_eq!(
            graph
                .unreachable
                .iter()
                .map(|(_, b)| b.label.clone())
                .collect::<Vec<_>>(),
            vec!["dead".to_string()]
        );
        assert_eq!(graph.blocks.len(), 2);
    }

    #[test]
    fn conflicting_register_types_are_rejected() {
        let f = FunctionBuilder::new("types")
            .block("entry")
            .instr(imm(1, 1))
            .instr(Instruction::Copy {
                ty: MirType::f64(),
                dst: gpr(1),
                src: Operand::imm_f64(1.0),
            })
            .instr(Instruction::Ret { value: None })
            .build();
        assert!(matches!(
            build(&f),
            Err(SsaBuildError::ConflictingTypes { .. })
        ));
    }

    #[test]
    fn single_definition_rule_rejects_missing_and_doubled_values() {
        let f = diamond();
        let mut graph = checked(&f);
        let join = graph
            .blocks
            .iter()
            .position(|b| b.label == "join")
            .expect("join");
        graph.blocks[join].insts[0].uses[0] = ValueId(99);
        let errs = graph.verify().unwrap_err();
        assert!(errs.iter().any(|e| e.rule == SsaRule::SingleDefinition));

        let mut graph = checked(&f);
        let (a, b) = (
            graph.blocks.iter().position(|x| x.label == "a").expect("a"),
            graph.blocks.iter().position(|x| x.label == "b").expect("b"),
        );
        let other = graph.blocks[b].insts[0].defines;
        graph.blocks[a].insts[0].defines = other;
        let errs = graph.verify().unwrap_err();
        assert!(errs.iter().any(|e| e.rule == SsaRule::SingleDefinition));
    }

    #[test]
    fn dominance_rule_rejects_use_of_a_sibling_arm_definition() {
        let f = diamond();
        let mut graph = checked(&f);
        let (a, b) = (
            graph.blocks.iter().position(|x| x.label == "a").expect("a"),
            graph.blocks.iter().position(|x| x.label == "b").expect("b"),
        );
        let sibling = graph.blocks[b].insts[0].defines.expect("b defines");
        graph.blocks[a].insts[0].uses.push(sibling);
        let errs = graph.verify().unwrap_err();
        assert!(errs.iter().any(|e| e.rule == SsaRule::Dominance));
    }

    #[test]
    fn dominance_rule_rejects_use_before_definition_in_block() {
        let f = FunctionBuilder::new("order")
            .block("entry")
            .instr(imm(1, 1))
            .instr(mov(2, 1))
            .instr(ret(2))
            .build();
        let mut graph = checked(&f);
        graph.blocks[1].insts[0].uses.push(ValueId(1));
        let errs = graph.verify().unwrap_err();
        assert!(errs.iter().any(|e| e.rule == SsaRule::Dominance));
    }

    #[test]
    fn edge_arity_rule_rejects_missing_argument() {
        let f = diamond();
        let mut graph = checked(&f);
        let a = graph.blocks.iter().position(|x| x.label == "a").expect("a");
        graph.blocks[a].edges[0].args.clear();
        let errs = graph.verify().unwrap_err();
        assert!(errs.iter().any(|e| e.rule == SsaRule::EdgeArity));
    }

    #[test]
    fn edge_types_rule_rejects_mismatched_parameter_type() {
        let f = diamond();
        let mut graph = checked(&f);
        let join = graph
            .blocks
            .iter()
            .position(|x| x.label == "join")
            .expect("join");
        let param = graph.blocks[join].params[0];
        graph.values[param.0 as usize].ty = Some(MirType::f64());
        let errs = graph.verify().unwrap_err();
        assert!(errs.iter().any(|e| e.rule == SsaRule::EdgeTypes));
    }
}
