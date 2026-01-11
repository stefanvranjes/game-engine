#pragma once

#include "PBDFluidSolver.h"
#include <vector>

#ifdef HAS_CUDA_TOOLKIT
#include <cuda_runtime.h>
#endif

class PBDFluidSolverGPU : public PBDFluidSolver {
public:
    PBDFluidSolverGPU();
    virtual ~PBDFluidSolverGPU();

    virtual void Initialize(PhysXBackend* backend) override;
    virtual void Update(std::vector<FluidParticle>& particles, 
                        const std::vector<FluidType>& fluidTypes,
                        float deltaTime) override;

private:
#ifdef HAS_CUDA_TOOLKIT
    // GPU Data Buffers
    FluidParticleGPU* d_Particles;
    int* d_SpatialIndices;    // Particle index
    int* d_SpatialHash;       // Hash value
    int* d_CellStart;         // Grid cell start index
    int* d_CellEnd;           // Grid cell end index
    
    // Neighbor List (Optional, can be computed on fly or stored)
    // For GPU, we usually use the grid directly in the solver kernel
    
    // Solver internal buffers (lambdas, deltas, etc.)
    float* d_Lambdas;
    float* d_DeltaPos; // usually x,y,z interleaved or separate
    
    // Capacity tracking to avoid reallocating every frame
    size_t m_GpuCapacity;
    
    // CUDA Streams
    cudaStream_t m_Stream;
    
    // Helper to resize GPU buffers if needed
    void ResizeBuffers(size_t numParticles);
    
    // Copy data
    void CopyToGPU(const std::vector<FluidParticle>& particles);
    void CopyFromGPU(std::vector<FluidParticle>& particles);
#endif
};
