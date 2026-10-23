use super::*;

type Position = (usize, usize);
type Expr = usize;

#[derive(Clone, PartialEq)]
enum Node {
    Value(Operand),
    Binary(IntBinOp, Expr, Expr),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Epoch {
    Old,
    Updated,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    register: Register,
    definition: Position,
    width: MirType,
    use_position: Position,
}

#[derive(Clone)]
pub(super) struct Polynomial {
    coefficients: [Expr; 3],
    pub updates: Vec<(Register, Position, Epoch)>,
}

impl Polynomial {
    fn degree(&self) -> usize {
        self.coefficients.iter().rposition(|&c| c != 0).unwrap_or(0)
    }
}

pub(super) struct Analysis<'a> {
    func: &'a Function,
    info: &'a LoopInfo,
    defs: &'a HashMap<Register, Vec<Position>>,
    product: Position,
    ty: MirType,
    nodes: Vec<Node>,
    memo: HashMap<Key, Option<Polynomial>>,
    active: HashSet<Key>,
    remaining: usize,
}

impl<'a> Analysis<'a> {
    pub fn new(
        func: &'a Function,
        info: &'a LoopInfo,
        defs: &'a HashMap<Register, Vec<Position>>,
        product: Position,
        ty: MirType,
    ) -> Self {
        Self {
            func,
            info,
            defs,
            product,
            ty,
            nodes: vec![Node::Value(Operand::Immediate(Immediate::I64(0)))],
            memo: HashMap::new(),
            active: HashSet::new(),
            remaining: 256,
        }
    }

    fn node(&mut self, node: Node) -> Option<Expr> {
        if let Some(index) = self.nodes.iter().position(|n| *n == node) {
            return Some(index);
        }
        if self.nodes.len() == 256 {
            return None;
        }
        self.nodes.push(node);
        Some(self.nodes.len() - 1)
    }

    fn value(&mut self, operand: Operand) -> Option<Expr> {
        if integer_constant(&operand) == Some(0) {
            Some(0)
        } else {
            self.node(Node::Value(operand))
        }
    }

    fn binary(&mut self, op: IntBinOp, a: Expr, b: Expr) -> Option<Expr> {
        match op {
            IntBinOp::Add if a == 0 => return Some(b),
            IntBinOp::Add | IntBinOp::Sub if b == 0 => return Some(a),
            IntBinOp::Mul if a == 0 || b == 0 => return Some(0),
            IntBinOp::Mul => {
                let one = |n: &Node| matches!(n, Node::Value(v) if integer_constant(v) == Some(1));
                if one(&self.nodes[a]) {
                    return Some(b);
                }
                if one(&self.nodes[b]) {
                    return Some(a);
                }
            }
            _ => {}
        }
        self.node(Node::Binary(op, a, b))
    }

    fn combine(&mut self, op: IntBinOp, a: Polynomial, b: Polynomial) -> Option<Polynomial> {
        let mut coefficients = [0; 3];
        match op {
            IntBinOp::Add | IntBinOp::Sub => {
                for (i, c) in coefficients.iter_mut().enumerate() {
                    *c = self.binary(op, a.coefficients[i], b.coefficients[i])?;
                }
            }
            IntBinOp::Mul if a.degree() + b.degree() <= 2 => {
                coefficients[0] = self.binary(op, a.coefficients[0], b.coefficients[0])?;
                if a.degree() == 0 || b.degree() == 0 {
                    for (i, c) in coefficients.iter_mut().enumerate().skip(1) {
                        *c = self.binary(op, a.coefficients[i], b.coefficients[0])?;
                        let other = self.binary(op, a.coefficients[0], b.coefficients[i])?;
                        *c = self.binary(IntBinOp::Add, *c, other)?;
                    }
                } else {
                    let ay = self.binary(op, a.coefficients[1], b.coefficients[0])?;
                    let bx = self.binary(op, b.coefficients[1], a.coefficients[0])?;
                    let ab = self.binary(op, a.coefficients[1], b.coefficients[1])?;
                    let sum = self.binary(IntBinOp::Add, ay, bx)?;
                    coefficients[1] = self.binary(IntBinOp::Add, sum, ab)?;
                    coefficients[2] = self.binary(IntBinOp::Add, ab, ab)?;
                }
            }
            _ => return None,
        }
        let mut updates = a.updates;
        for update in b.updates {
            if !updates.contains(&update) {
                updates.push(update);
            }
        }
        Some(Polynomial {
            coefficients,
            updates,
        })
    }

    fn operand(&mut self, operand: &Operand, use_position: Position) -> Option<Polynomial> {
        self.remaining = self.remaining.checked_sub(1)?;
        if invariant(operand, self.defs) {
            return Some(Polynomial {
                coefficients: [self.value(operand.clone())?, 0, 0],
                updates: Vec::new(),
            });
        }
        let Operand::Register(register) = operand else {
            return None;
        };
        if !register.is_virtual() {
            return None;
        }
        let definitions = self.defs.get(register)?;
        if definitions.len() != 1 {
            return None;
        }
        let definition = definitions[0];
        let key = Key {
            register: register.clone(),
            definition,
            width: self.ty,
            use_position,
        };
        if let Some(result) = self.memo.get(&key) {
            return result.clone();
        }
        if !self.active.insert(key.clone()) {
            return None;
        }
        let result = self.definition(register, definition, use_position);
        self.active.remove(&key);
        self.memo.insert(key, result.clone());
        result
    }

    fn definition(
        &mut self,
        register: &Register,
        definition: Position,
        use_position: Position,
    ) -> Option<Polynomial> {
        let instruction = &self.func.blocks[definition.0].instructions[definition.1];
        if let Instruction::IntBinary {
            op: IntBinOp::Add,
            ty,
            lhs,
            rhs,
            ..
        } = instruction
            && *ty == self.ty
        {
            let step = match (lhs, rhs) {
                (Operand::Register(r), step) | (step, Operand::Register(r))
                    if r == register && invariant(step, self.defs) =>
                {
                    Some(step)
                }
                _ => None,
            };
            if let Some(step) = step {
                let epoch = if definition.0 == use_position.0 {
                    if definition.1 < use_position.1 {
                        Epoch::Updated
                    } else {
                        Epoch::Old
                    }
                } else {
                    let dominators = calculate_dominators(self.func);
                    if dominators
                        .get(&self.func.blocks[use_position.0].label)
                        .is_some_and(|d| d.contains(&self.func.blocks[definition.0].label))
                    {
                        Epoch::Updated
                    } else {
                        Epoch::Old
                    }
                };
                let initial = self.value(Operand::Register(register.clone()))?;
                let delta = self.value(step.clone())?;
                return Some(Polynomial {
                    coefficients: [initial, delta, 0],
                    updates: vec![(register.clone(), definition, epoch)],
                });
            }
        }
        // Only definitions in the product's straight-line prefix can be expanded.
        if definition.0 != self.product.0
            || definition.0 != use_position.0
            || definition.1 >= use_position.1
            || !self
                .info
                .blocks
                .contains(&self.func.blocks[definition.0].label)
        {
            return None;
        }
        match instruction {
            Instruction::Copy { ty, src, .. } if *ty == self.ty => self.operand(src, definition),
            Instruction::IntBinary {
                op, ty, lhs, rhs, ..
            } if *ty == self.ty => {
                let op = *op;
                let a = self.operand(lhs, definition)?;
                let b = self.operand(rhs, definition)?;
                self.combine(op, a, b)
            }
            _ => None,
        }
    }

    pub fn product(&mut self, lhs: &Operand, rhs: &Operand) -> Option<Polynomial> {
        let a = self.operand(lhs, self.product)?;
        let b = self.operand(rhs, self.product)?;
        let both_vary = !a.updates.is_empty() && !b.updates.is_empty();
        let polynomial = self.combine(IntBinOp::Mul, a, b)?;
        ((polynomial.degree() == 2 || both_vary)
            && polynomial.updates.iter().all(|u| u.2 == Epoch::Old))
        .then_some(polynomial)
    }

    pub fn emit(
        &self,
        polynomial: &Polynomial,
        next_id: &mut u32,
        instructions: &mut Vec<Instruction>,
    ) -> Option<[Operand; 3]> {
        fn materialize(
            analysis: &Analysis<'_>,
            expr: Expr,
            next_id: &mut u32,
            instructions: &mut Vec<Instruction>,
            memo: &mut HashMap<Expr, Operand>,
        ) -> Option<Operand> {
            if let Some(value) = memo.get(&expr) {
                return Some(value.clone());
            }
            let value = match &analysis.nodes[expr] {
                Node::Value(v) => v.clone(),
                Node::Binary(op, a, b) => {
                    let lhs = materialize(analysis, *a, next_id, instructions, memo)?;
                    let rhs = materialize(analysis, *b, next_id, instructions, memo)?;
                    let dst: Register = VirtualReg::gpr(*next_id).into();
                    *next_id = next_id.checked_add(1)?;
                    instructions.push(Instruction::IntBinary {
                        op: *op,
                        dst: dst.clone(),
                        ty: analysis.ty,
                        lhs,
                        rhs,
                    });
                    Operand::Register(dst)
                }
            };
            memo.insert(expr, value.clone());
            Some(value)
        }
        let mut memo = HashMap::new();
        let [p, d, e] = polynomial.coefficients;
        Some([
            materialize(self, p, next_id, instructions, &mut memo)?,
            materialize(self, d, next_id, instructions, &mut memo)?,
            materialize(self, e, next_id, instructions, &mut memo)?,
        ])
    }
}
