// FEAT: TODO: Automated Register Allocation 
// | Currently I have saw your register allocation implementation 
// | and for future extendability, I think making a Register Allocator as a Trait
// | and then implement it for each Arch/Target would be a good idea.
// | How do you think about this?
// | on arm/aarch64 I have my version of the register allocator which is quite simple

pub trait RegisterAllocator {