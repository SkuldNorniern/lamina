#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuIntrinsic {
    ThreadId(GpuAxis),
    BlockId(GpuAxis),
    BlockDim(GpuAxis),
    GridDim(GpuAxis),
    LaneId,
    WorkgroupBarrier,
}

impl GpuIntrinsic {
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.strip_prefix('@').unwrap_or(name);
        match name {
            "__lamina_gpu_lane_id" => return Some(Self::LaneId),
            "__lamina_gpu_workgroup_barrier" => return Some(Self::WorkgroupBarrier),
            _ => {}
        }
        let (operation, axis) = name.rsplit_once('_')?;
        let axis = match axis {
            "x" => GpuAxis::X,
            "y" => GpuAxis::Y,
            "z" => GpuAxis::Z,
            _ => return None,
        };
        match operation {
            "__lamina_gpu_thread_id" => Some(Self::ThreadId(axis)),
            "__lamina_gpu_block_id" => Some(Self::BlockId(axis)),
            "__lamina_gpu_block_dim" => Some(Self::BlockDim(axis)),
            "__lamina_gpu_grid_dim" => Some(Self::GridDim(axis)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_calls_are_exact() {
        assert_eq!(
            GpuIntrinsic::from_name("@__lamina_gpu_thread_id_x"),
            Some(GpuIntrinsic::ThreadId(GpuAxis::X))
        );
        assert_eq!(
            GpuIntrinsic::from_name("__lamina_gpu_workgroup_barrier"),
            Some(GpuIntrinsic::WorkgroupBarrier)
        );
        assert_eq!(GpuIntrinsic::from_name("__lamina_gpu_thread_id_w"), None);
        assert_eq!(GpuIntrinsic::from_name("print"), None);
    }
}
