# Proposal: Native MX Format Support in XLA

## Executive Summary

This document proposes an architecture for supporting MX (Microscaling) formats like MXFP8 as native data types in XLA, enabling users to write JAX models with `dtype=mxfp8` and have all operations automatically work with full fusion support, avoiding materialization of dequantized tensors in HBM.

## Background

### What are MX Formats?

MX (Microscaling) formats are block-scaled quantization formats standardized by the Open Compute Project (OCP). Key properties:
- **Block size**: Typically 32 elements per block
- **Compound structure**: Each block contains quantized values + shared scale factor
- **MXFP8 variants**:
  - MXFP8-E4M3: F8E4M3FN values + F8E8M0FNU scale (32 values + 1 scale = 33 bytes)
  - MXFP8-E5M2: F8E5M2 values + F8E8M0FNU scale

### Current XLA Support

XLA already has **partial** MX support:

1. **Primitive types** (`xla/xla_data.proto:32,33`):
   - `F8E4M3FN`, `F8E5M2` (value types)
   - `F8E8M0FNU` (scale type)
   - `F4E2M1FN` (NVFP4)

2. **Block scaling infrastructure** (`xla/service/gpu/transforms/block_scaling_rewriter.h`):
   - Custom calls: `__op$quantize`, `__op$dequantize`, `__op$block_scaled_dot`
   - Block size constants: `kBlockSizeMXFP8 = 32`, `kBlockSizeNVFP4 = 16`

3. **HloScaledDotInstruction** (`xla/hlo/ir/hlo_instructions.h:2721`):
   - 4 operands: `lhs, rhs, lhs_scale, rhs_scale`
   - First-class HLO opcode: `kScaledDot`

4. **Triton fusion support** (`xla/service/gpu/triton_fusion_analysis.h:56-62`):
   - Scale scopes: `LHS_SCALE`, `RHS_SCALE`, `OUTPUT`
   - Block size tracking: `lhs_scale_block_size()`, `rhs_scale_block_size()`

5. **cuDNN kernels** (`xla/service/gpu/transforms/block_scaling_rewriter.cc:254-313`):
   - Native MXFP8 matmul support
   - Mixed precision: E4M3×E4M3, E4M3×E5M2, E5M2×E4M3

### The Problem

Current representation is **not native**:
- MX tensors are TWO separate tensors: `(quantized_values, scales)`
- Requires explicit custom calls to quantize/dequantize
- Can't write `jnp.matmul(a, b)` where `a.dtype == mxfp8`
- Pattern matching needed to fuse quantization

The fundamental challenge: **MX formats are compound types** - they don't fit into XLA's uniform primitive type system.

## Proposed Architecture

### Core Concept: Layout-Based Encoding

Instead of creating a new primitive type `MXFP8`, represent MX tensors as:
```
dtype = F8E4M3FN (or F8E5M2)
layout = MX_BLOCK_SCALED {
  block_size: 32,
  scale_dtype: F8E8M0FNU,
  axis: -1  // which axis is block-scaled
}
```

This separates:
- **Value type** (F8E4M3FN) - what each element represents
- **Encoding** (MX_BLOCK_32) - how it's physically stored and accessed

### Key Components

#### 1. Extended Layout System

**File**: `xla/shape.proto`, `xla/layout.h`

Add new layout type:
```protobuf
message Layout {
  // ... existing fields ...

  optional BlockScaledLayout block_scaled_layout = 10;
}

message BlockScaledLayout {
  // Block size (e.g., 32 for MXFP8)
  int64 block_size = 1;

  // Scale factor dtype (e.g., F8E8M0FNU)
  PrimitiveType scale_dtype = 2;

  // Which dimension is block-scaled (typically -1 = last dim)
  int64 scaled_dimension = 3;

  // Layout of scale tensor
  Layout scale_layout = 4;
}
```

**Rationale**: Layouts already describe physical memory representation. Block scaling is fundamentally a memory layout concern.

#### 2. Shape Utilities for MX Tensors

**File**: `xla/shape_util.h`

Add helper functions:
```cpp
// Check if shape uses MX encoding
bool ShapeUtil::IsBlockScaled(const Shape& shape);

// Get the shape of the scale tensor
Shape ShapeUtil::GetScaleShape(const Shape& block_scaled_shape);

// Create MX-encoded shape
Shape ShapeUtil::MakeMXShape(
    PrimitiveType value_type,
    PrimitiveType scale_type,
    absl::Span<const int64_t> dimensions,
    int64_t block_size,
    int64_t scaled_dimension);
```

#### 3. Buffer Allocation

**File**: `xla/service/gpu/gpu_executable.cc`

Modify buffer assignment to:
1. Detect MX layouts
2. Allocate **two adjacent buffers**: values buffer + scales buffer
3. Store metadata mapping value buffer → scale buffer

```cpp
struct MXBufferPair {
  BufferAllocation values;
  BufferAllocation scales;
};

class BufferAssignment {
  // Map from MX value buffer index to scale buffer
  absl::flat_hash_map<BufferAllocation::Index, BufferAllocation::Index>
      mx_scale_buffers_;
};
```

**Key insight**: The two buffers are **always allocated together**, managed as a unit, but can be independently accessed for optimization.

#### 4. HLO Instruction Support

**File**: `xla/hlo/ir/hlo_instruction.cc`

Extend existing ops to work with MX layouts:

**a) Dot/Matmul**: Automatically lower to `ScaledDot`
```cpp
// In HloDotInstruction::CreateFromProto or similar
if (ShapeUtil::IsBlockScaled(lhs_shape) ||
    ShapeUtil::IsBlockScaled(rhs_shape)) {
  // Extract scale buffers
  HloInstruction* lhs_scale = GetScaleBuffer(lhs);
  HloInstruction* rhs_scale = GetScaleBuffer(rhs);

  // Create ScaledDot instead of regular Dot
  return HloInstruction::CreateScaledDot(...);
}
```

**b) Element-wise ops**: Operate on both buffers
```cpp
// For add(mx_a, mx_b):
// 1. Fuse scale adjustment into computation
// 2. Result uses max scale: scale_out = max(scale_a, scale_b)
// 3. Adjust values: value_out = value_a * (scale_a / scale_out) +
//                                value_b * (scale_b / scale_out)
```

**c) Conversions**: Add quantize/dequantize
```cpp
// Convert F32 -> MXFP8: quantize
// Convert MXFP8 -> F32: dequantize
HloInstruction* CreateConvert(HloInstruction* operand,
                              const Shape& target_shape) {
  if (IsBlockScaled(operand->shape()) &&
      !IsBlockScaled(target_shape)) {
    // Dequantize
    return CreateCustomCall("__op$dequantize", ...);
  }
  if (!IsBlockScaled(operand->shape()) &&
      IsBlockScaled(target_shape)) {
    // Quantize
    return CreateCustomCall("__op$quantize", ...);
  }
}
```

#### 5. Fusion Integration

**File**: `xla/service/gpu/transforms/gemm_fusion.cc`

Extend GEMM fusion to automatically handle MX formats:

```cpp
class GemmFusion {
  absl::StatusOr<HloInstruction*> FuseGemm(HloInstruction* dot) {
    // Check if inputs are MX-encoded
    bool lhs_is_mx = ShapeUtil::IsBlockScaled(lhs->shape());
    bool rhs_is_mx = ShapeUtil::IsBlockScaled(rhs->shape());

    if (lhs_is_mx || rhs_is_mx) {
      // Automatically extract scales and create ScaledDot fusion
      return CreateMXScaledDotFusion(dot);
    }
  }

  // Fuse element-wise epilogue with scale management
  void FuseEpilogue(HloFusionInstruction* fusion) {
    // For MX outputs, incorporate scale calculation into fusion
    // Avoid materializing dequantized values
  }
};
```

**Key**: Pattern match and rewrite to avoid dequantization:
```
Before (naive):
  dequant(mx_a) -> matmul -> add -> quant(mx_out)

After (optimized):
  scaled_dot(mx_a.values, mx_a.scales, ...)
    -> fused_add_with_scale_adjust
    -> (mx_out.values, mx_out.scales)
```

#### 6. Triton Code Generation

**File**: `xla/backends/gpu/codegen/triton/fusion_emitter.cc`

Extend Triton emitter to generate MX-aware kernels:

```cpp
mlir::Value EmitMXLoad(const HloInstruction* param,
                       const BlockLevelParameters& params) {
  // Load values
  mlir::Value values = EmitLoad(param->values_buffer());

  // Load corresponding scales (coalesced)
  int block_size = GetMXBlockSize(param->shape());
  mlir::Value scales = EmitBlockScaleLoad(
      param->scale_buffer(), block_size);

  // Emit dequantization inline (fuses with compute)
  // For F8E4M3 * scale: use efficient fp8->fp32 + multiply
  return CreateDequantizeOp(values, scales);
}

void EmitMXDot(HloScaledDotInstruction* dot) {
  // Use native Triton/PTX instructions for scaled matmul
  // E.g., HMMA with scale factors
  // or WGMMA with MX format support
}
```

**Hardware utilization**:
- NVIDIA Hopper+: Native MX support via WGMMA instructions
- Older GPUs: Efficient fused dequant + HMMA

#### 7. Python/JAX Frontend

**File**: `xla/python/types.cc` (XLA side), JAX repo (user-facing)

Add dtype registration:
```python
# In JAX
import jax.numpy as jnp

# Register MX dtypes
mxfp8_e4m3 = jnp.dtype('mxfp8_e4m3')  # Maps to F8E4M3FN with MX layout
mxfp8_e5m2 = jnp.dtype('mxfp8_e5m2')  # Maps to F8E5M2 with MX layout

# User code - just works!
a = jnp.array(data, dtype=mxfp8_e4m3)
b = jnp.array(weights, dtype=mxfp8_e4m3)
c = jnp.matmul(a, b)  # Compiles to ScaledDot, fully fused
c = jnp.tanh(c)       # Fused into ScaledDot epilogue
```

**XLA integration**:
```cpp
// In xla/python/types.cc
Shape ConvertDtype(py::dtype dtype, absl::Span<int64_t> dims) {
  if (dtype == "mxfp8_e4m3") {
    return ShapeUtil::MakeMXShape(
        F8E4M3FN, F8E8M0FNU, dims,
        /*block_size=*/32, /*scaled_dim=*/-1);
  }
}
```

### End-to-End Flow Example

**User code**:
```python
import jax.numpy as jnp

def model(x, w):
  return jnp.tanh(jnp.matmul(x, w))

# Specify MX format
x = jnp.array(input_data, dtype=jnp.mxfp8_e4m3)
w = jnp.array(weights, dtype=jnp.mxfp8_e4m3)
y = model(x, w)
```

**XLA compilation flow**:

1. **Frontend**:
   - JAX sees `dtype=mxfp8_e4m3`
   - Sends to XLA: `Shape{F8E4M3FN, layout=MX_BLOCK_32}`

2. **Buffer Assignment**:
   - Allocates value buffer: `[M, N] of F8E4M3FN`
   - Allocates scale buffer: `[M, N/32] of F8E8M0FNU`
   - Links them in metadata

3. **HLO Construction**:
   - `matmul(x, w)` with MX inputs
   - Detects MX layout
   - Automatically creates: `ScaledDot(x.values, w.values, x.scales, w.scales)`

4. **Fusion**:
   - GEMM fusion detects `ScaledDot`
   - Fuses `tanh` into epilogue
   - Creates Triton fusion: `ScaledDot + tanh(result)`

5. **Code Generation**:
   - Triton emitter generates:
     - Loads for values and scales
     - Inline dequantization (fused with matmul)
     - Matmul using F8 tensor cores
     - Fused tanh on accumulator
     - Inline quantization of output
     - Store values and scales

6. **Execution**:
   - Single kernel launch
   - **No materialization** of F32/BF16 tensors in HBM
   - All quantization/dequantization in registers/shared memory

## Implementation Phases

### Phase 1: Core Infrastructure (4-6 weeks)

1. **Layout system extensions**
   - Add `BlockScaledLayout` to protobuf
   - Implement `ShapeUtil::IsBlockScaled()` utilities
   - Add layout validation

2. **Buffer allocation**
   - Extend `BufferAssignment` for MX pairs
   - Implement scale buffer tracking
   - Update buffer aliasing logic

3. **Basic conversions**
   - Implement explicit quantize/dequantize ops
   - Add shape inference for MX formats
   - Test round-trip conversions

**Deliverable**: Can create MX tensors and convert to/from F32

### Phase 2: Dot/Matmul Support (3-4 weeks)

1. **Automatic ScaledDot lowering**
   - Detect MX inputs in `HloDotInstruction`
   - Auto-create `HloScaledDotInstruction`
   - Extract scale buffers

2. **cuDNN integration**
   - Leverage existing cuDNN MX kernels
   - Add layout-aware kernel selection
   - Benchmarking

3. **Testing**
   - Correctness tests against F32 reference
   - Performance benchmarks vs explicit quantization

**Deliverable**: `jnp.matmul(mx_a, mx_b)` works and is fast

### Phase 3: Fusion Support (4-6 weeks)

1. **GEMM fusion for MX**
   - Extend `GemmFusion` pattern matching
   - Handle MX in `TritonFusionAnalysis`
   - Epilogue fusion (element-wise ops)

2. **Triton code generation**
   - MX load/store emission
   - Inline dequantization
   - Scale calculation for outputs

3. **Optimization**
   - Avoid redundant quantization
   - Scale caching in shared memory
   - Kernel tuning

**Deliverable**: Full fusion working, no HBM dequantization

### Phase 4: Element-wise Operations (3-4 weeks)

1. **Element-wise op support**
   - Add, multiply, ReLU, tanh, etc.
   - Scale adjustment logic
   - Fusion with other ops

2. **Broadcasting**
   - Handle broadcast of MX tensors
   - Scale broadcast rules

3. **Reductions**
   - Sum, max, etc. with MX inputs
   - Output scale calculation

**Deliverable**: Most JAX ops work with MX dtypes

### Phase 5: Frontend & Polish (2-3 weeks)

1. **JAX integration**
   - Register `mxfp8_e4m3`, `mxfp8_e5m2` dtypes
   - Type promotion rules
   - Documentation

2. **Profiling & debugging**
   - Add MX-aware profiling
   - Visualization of scale tensors
   - Debug assertions

3. **Testing & benchmarks**
   - End-to-end model tests
   - Performance benchmarks
   - Comparison with manual quantization

**Deliverable**: Production-ready MX support in JAX

## Alternative Approaches Considered

### Alternative 1: Packed Representation

**Idea**: Store MX data as single buffer with interleaved scales.

**Layout**: `[v0...v31, s0, v32...v63, s1, ...]`

**Pros**:
- Single buffer, simpler allocation
- True "native" type

**Cons**:
- Non-coalesced memory access (scales not contiguous)
- Harder to optimize (can't separate value/scale access patterns)
- Complex indexing
- **Rejected**: Performance would be poor

### Alternative 2: New Primitive Type

**Idea**: Add `MXFP8` as primitive type like `F32`, `BF16`.

**Pros**:
- Most "native" feeling
- Cleaner abstraction

**Cons**:
- **Massive type system changes** - every pass needs updates
- Shape inference becomes complex (implicit dual buffers)
- Doesn't generalize (what about other block sizes?)
- **Rejected**: Too invasive, doesn't match reality of compound data

### Alternative 3: Explicit Tuple Types

**Idea**: MX tensor is `Tuple(values: F8E4M3FN, scales: F8E8M0FNU)`.

**Pros**:
- Explicit representation
- Type system already handles tuples

**Cons**:
- User must explicitly manage tuple components
- Every op needs tuple wrapping/unwrapping
- Lots of boilerplate
- **Rejected**: Not ergonomic, defeats "native" goal

### Why Layout-Based is Best

1. **Separation of concerns**: Value type vs. encoding
2. **Builds on existing infrastructure**: Layouts, ScaledDot, cuDNN kernels
3. **Flexible**: Easy to add other block sizes, formats
4. **Performant**: Can optimize value/scale access independently
5. **Pragmatic**: Incremental implementation path

## Performance Considerations

### Memory Efficiency

**Savings**: MXFP8 vs BF16
- BF16: 2 bytes/element
- MXFP8: (32×1 + 1×1) / 32 = 1.03125 bytes/element
- **50% memory reduction**

### Compute Efficiency

**Tensor Core Utilization**:
- Hopper (H100): Native MXFP8 via WGMMA
- Ampere/Ada (A100, RTX 4090): FP8 tensor cores + scale multiplication
- **2-4x speedup** over BF16 (depending on memory-bound vs compute-bound)

### Fusion Benefits

**Without fusion** (naive):
```
Load BF16 -> Quantize to MXFP8 (HBM write) ->
  Load MXFP8 (HBM read) -> Matmul ->
  Dequantize to BF16 (HBM write) -> Load BF16 (HBM read)
```
**4 extra HBM accesses per tensor** = 2x BF16 traffic!

**With fusion** (proposed):
```
Load MXFP8 (values + scales) -> Matmul (dequant in SHMEM) ->
  Epilogue (requant in SHMEM) -> Store MXFP8
```
**Actual memory savings**: 50% of BF16 baseline

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Type system complexity | High | Phase implementation, extensive testing |
| Backend compatibility | Medium | Fallback to dequant for older GPUs |
| Numerical accuracy | Medium | Rigorous testing, configurable scales |
| Debuggability | Low | Add MX-aware visualization tools |
| JAX API changes | Low | Additive changes only, backward compatible |

## Success Metrics

1. **Functionality**: 90%+ of JAX ops work with MX dtypes
2. **Performance**: Within 5% of manually-optimized MX kernels
3. **Memory**: 50% reduction vs BF16 (theoretical limit)
4. **Usability**: 1-line dtype change in user code
5. **Fusion**: No unnecessary HBM dequantization in common patterns

## References

- [OCP Microscaling Formats Spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/pdf/2310.10537)
- XLA Block Scaling Rewriter: `xla/service/gpu/transforms/block_scaling_rewriter.cc`
- XLA ScaledDot: `xla/hlo/ir/hlo_instructions.h:2721`
- Triton Fusion Analysis: `xla/service/gpu/triton_fusion_analysis.h`

## Conclusion

The proposed layout-based approach provides:
- ✅ **Native feel**: Users just specify `dtype=mxfp8`
- ✅ **Performance**: Full fusion, no HBM dequantization
- ✅ **Pragmatic**: Builds on existing infrastructure
- ✅ **Flexible**: Generalizes to other MX formats
- ✅ **Achievable**: Clear implementation path

This architecture makes MX formats truly first-class in XLA while avoiding the pitfalls of more invasive approaches.
