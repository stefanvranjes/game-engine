#include "PBDFluidSolverGPU.h"
#include <iostream>
#include <algorithm>

#ifdef HAS_CUDA_TOOLKIT
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// --- CUDA Helper Macros ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while (0)

// --- Constants ---
__constant__ float c_KernelRadius;
__constant__ float c_Poly6Coeff;
__constant__ float c_SpikyGradCoeff;
__constant__ float c_ViscosityLapCoeff;
__constant__ Vec3 c_Gravity;
__constant__ Vec3 c_BoundaryMin;
__constant__ Vec3 c_BoundaryMax;
__constant__ bool c_EnableBoundary;

// --- Kernels ---

// 1. Predict Positions
__global__ void PredictPositionsKernel(
    FluidParticleGPU* particles, 
    int numParticles, 
    float dt) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    FluidParticleGPU& p = particles[idx];
    
    // Semi-implicit Euler integration
    // v = v + g * dt
    p.velocity[0] += c_Gravity.x * dt;
    p.velocity[1] += c_Gravity.y * dt;
    p.velocity[2] += c_Gravity.z * dt;

    // p = p + v * dt
    p.predictedPosition[0] = p.position[0] + p.velocity[0] * dt;
    p.predictedPosition[1] = p.position[1] + p.velocity[1] * dt;
    p.predictedPosition[2] = p.position[2] + p.velocity[2] * dt;
    
    // Boundary handling (simple clamp/bounce prediction)
    if (c_EnableBoundary) {
        if (p.predictedPosition[1] < c_BoundaryMin.y) {
            p.predictedPosition[1] = c_BoundaryMin.y;
            p.velocity[1] = 0.0f; // Kill velocity on floor for stability during prediction
        }
        // ... add other boundaries as essential ...
    }
}

// 2. Spatial Hashing
__device__ int GetGridHash(int x, int y, int z) {
    // Simple hash (Prime numbers)
    // Warning: Map infinite space to finite hash table size if using hash map.
    // For Sort-based, we map to linear index or full hash. 
    // Here using linear index assuming bounded domain or hash collisions allowed.
    // Better: ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) % numCells
    return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791));
}

__device__ int3 GetGridPos(float x, float y, float z, float cellSize) {
    int3 pos;
    pos.x = floorf(x / cellSize);
    pos.y = floorf(y / cellSize);
    pos.z = floorf(z / cellSize);
    return pos;
}

__global__ void CalcHashKernel(
    FluidParticleGPU* particles,
    int numParticles,
    int* spatialIndices,
    int* spatialHash,
    float cellSize,
    int numCells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    FluidParticleGPU& p = particles[idx];
    
    int3 gridPos = GetGridPos(p.predictedPosition[0], p.predictedPosition[1], p.predictedPosition[2], cellSize);
    int hash = GetGridHash(gridPos.x, gridPos.y, gridPos.z);
    
    // Constrain hash to array size
    hash = abs(hash) % numCells;
    
    spatialIndices[idx] = idx;
    spatialHash[idx] = hash;
}

__global__ void ReorderParticlesKernel(
    int numParticles,
    int* spatialIndices,
    FluidParticleGPU* sortedParticles,
    FluidParticleGPU* unsortedParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    int sortedIdx = idx; // We are iterating the sorted buffer
    int originalIdx = spatialIndices[idx]; // Index of particle that should go here
    
    // Copy data
    // Usually easier to sort the indices and access indirectly, 
    // but reordering memory improves cache coherency.
    // For now, let's just keep indirect access in solver or reorder. 
    // Reordering is better for coalescing.
    sortedParticles[idx] = unsortedParticles[originalIdx];
}

__global__ void FindCellStartEndKernel(
    int numParticles,
    int* spatialHash,
    int* cellStart,
    int* cellEnd,
    int numCells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int hash = spatialHash[idx];
    
    if (idx == 0) {
        cellStart[hash] = idx;
    } else {
        int prevHash = spatialHash[idx - 1];
        if (hash != prevHash) {
            cellEnd[prevHash] = idx;
            cellStart[hash] = idx;
        }
    }
    
    if (idx == numParticles - 1) {
        cellEnd[hash] = numParticles;
    }
}

// 3. Density & Lambda
__device__ float Poly6KernelGPU(float r2, float h) {
    float h2 = h * h;
    if (r2 >= 0 && r2 <= h2) {
        return c_Poly6Coeff * powf(h2 - r2, 3);
    }
    return 0.0f;
}

__device__ Vec3 SpikyGradientGPU(const Vec3& rVec, float h) {
    float r = rVec.Length();
    if (r > 0 && r <= h) {
        float factor = c_SpikyGradCoeff * powf(h - r, 2) / r; // Normalize direction
        return rVec * (-factor); // Direction * factor (negated because gradient is -) -- Wait, Spiky is positive, gradient points away? No, towards center is negative slope.
        // Formula: -45 / (pi * h^6) * (h-r)^2 * r_unit
        // r_unit = rVec / r
        // Result = -k * (h-r)^2 * rVec / r
    }
    return Vec3(0,0,0);
}

__global__ void ComputeDensityPressureKernel(
    FluidParticleGPU* particles,
    int numParticles,
    int* cellStart,
    int* cellEnd,
    int numCells,
    float cellSize,
    float restDensity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    FluidParticleGPU& p_i = particles[idx];
    Vec3 pos_i(p_i.predictedPosition[0], p_i.predictedPosition[1], p_i.predictedPosition[2]);
    
    float density = 0.0f;
    
    // Neighbor search
    int3 gridPos = GetGridPos(pos_i.x, pos_i.y, pos_i.z, cellSize);
    
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int hash = GetGridHash(gridPos.x + x, gridPos.y + y, gridPos.z + z);
                hash = abs(hash) % numCells;
                
                int start = cellStart[hash];
                int end = cellEnd[hash];
                
                if (start == -1) continue; // Empty cell
                
                for (int j = start; j < end; j++) {
                    FluidParticleGPU& p_j = particles[j];
                    
                    Vec3 pos_j(p_j.predictedPosition[0], p_j.predictedPosition[1], p_j.predictedPosition[2]);
                    Vec3 diff = pos_i - pos_j;
                    float r2 = diff.LengthSquared();
                    
                    if (r2 <= c_KernelRadius * c_KernelRadius) {
                        density += p_j.mass * Poly6KernelGPU(r2, c_KernelRadius);
                    }
                }
            }
        }
    }
    
    p_i.density = density;
    float C = density / restDensity - 1.0f;
    
    // Lambda (Constraint Scaling Factor)
    // lambda_i = -C_i / (sum |grad C_j|^2)
    // simplified: see PBD paper
    
    // Not implementing full lambda calculation here yet, just density for now.
    // In PBD, we need gradients to compute lambda.
}

__global__ void SolveDensityConstraintsKernel(
    FluidParticleGPU* particles,
    int numParticles,
    int* cellStart,
    int* cellEnd,
    int numCells,
    float cellSize,
    float restDensity,
    float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    FluidParticleGPU& p_i = particles[idx];
    // Calculate Lambda
    // ... (Implementation of PBD Lambda and DeltaPos) 
    // For brevity, using simple PBD density correction placeholder logic
    // Real implementation would duplicate PBDFluidSolver::ComputeLambda & ComputeDeltaPos logic
    // but on GPU.
}


// --- Class Implementation ---

PBDFluidSolverGPU::PBDFluidSolverGPU() 
    : d_Particles(nullptr)
    , d_SpatialIndices(nullptr)
    , d_SpatialHash(nullptr)
    , d_CellStart(nullptr)
    , d_CellEnd(nullptr)
    , d_Lambdas(nullptr)
    , d_DeltaPos(nullptr)
    , m_GpuCapacity(0)
    , m_Stream(nullptr)
{
    cudaStreamCreate(&m_Stream);
    std::cout << "PBDFluidSolverGPU Created" << std::endl;
}

PBDFluidSolverGPU::~PBDFluidSolverGPU() {
    if (d_Particles) cudaFree(d_Particles);
    if (d_SpatialIndices) cudaFree(d_SpatialIndices);
    if (d_SpatialHash) cudaFree(d_SpatialHash);
    if (d_CellStart) cudaFree(d_CellStart);
    if (d_CellEnd) cudaFree(d_CellEnd);
    if (d_Lambdas) cudaFree(d_Lambdas);
    if (d_DeltaPos) cudaFree(d_DeltaPos);
    
    cudaStreamDestroy(m_Stream);
}

void PBDFluidSolverGPU::Initialize(PhysXBackend* backend) {
    PBDFluidSolver::Initialize(backend);
    
    // Copy constants to GPU
    // float h = GetKernelRadius();
    // CUDA_CHECK(cudaMemcpyToSymbol(c_KernelRadius, &h, sizeof(float)));
    // ... copy other constants ...
}

void PBDFluidSolverGPU::ResizeBuffers(size_t numParticles) {
    if (numParticles > m_GpuCapacity) {
        size_t newCap = numParticles * 1.5;
        
        // Free old
        if (d_Particles) cudaFree(d_Particles);
        if (d_SpatialIndices) cudaFree(d_SpatialIndices);
        if (d_SpatialHash) cudaFree(d_SpatialHash);
        
        // Allocate new
        CUDA_CHECK(cudaMalloc(&d_Particles, newCap * sizeof(FluidParticleGPU)));
        CUDA_CHECK(cudaMalloc(&d_SpatialIndices, newCap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_SpatialHash, newCap * sizeof(int)));
        
        // Grid size is fixed or dynamic? Fixed for now
        int numCells = 100000; // Example
        if (!d_CellStart) {
            CUDA_CHECK(cudaMalloc(&d_CellStart, numCells * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_CellEnd, numCells * sizeof(int)));
        }
        
        m_GpuCapacity = newCap;
    }
}

void PBDFluidSolverGPU::Update(std::vector<FluidParticle>& particles, 
                               const std::vector<FluidType>& fluidTypes,
                               float deltaTime) 
{
    int numParticles = particles.size();
    if (numParticles == 0) return;
    
    ResizeBuffers(numParticles);
    
    // 1. Upload to GPU
    // Optimize: only upload changed data / use mapped memory
    CopyToGPU(particles);
    
    // 2. Set Constants
    float h = GetKernelRadius();
    CUDA_CHECK(cudaMemcpyToSymbol(c_KernelRadius, &h, sizeof(float)));
    Vec3 g = GetGravity();
    CUDA_CHECK(cudaMemcpyToSymbol(c_Gravity, &g, sizeof(Vec3)));
    
    // 3. Run Kernels
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    
    PredictPositionsKernel<<<numBlocks, blockSize, 0, m_Stream>>>(d_Particles, numParticles, deltaTime);
    
    // Spatial Hashing
    int numCells = 100000;
    float cellSize = h;
    CalcHashKernel<<<numBlocks, blockSize, 0, m_Stream>>>(d_Particles, numParticles, d_SpatialIndices, d_SpatialHash, cellSize, numCells);
    
    // Sort (using Thrust)
    thrust::device_ptr<int> t_spatialHash(d_SpatialHash);
    thrust::device_ptr<int> t_spatialIndices(d_SpatialIndices);
    thrust::sort_by_key(t_spatialHash, t_spatialHash + numParticles, t_spatialIndices);
    
    // Reset Cells
    CUDA_CHECK(cudaMemsetAsync(d_CellStart, -1, numCells * sizeof(int), m_Stream));
    
    // Find Cell Starts
    FindCellStartEndKernel<<<numBlocks, blockSize, 0, m_Stream>>>(numParticles, d_SpatialHash, d_CellStart, d_CellEnd, numCells);
    
    // TODO: Reorder particles buffer for coalescing (Optional but good)
    
    // Constraints (Iterations)
    for (int iter = 0; iter < GetSolverIterations(); ++iter) {
         // Solve Density
         // SolveDensityConstraintsKernel<<<...>>>
    }
    
    // Update Velocity & Position
    // UpdateVelocitiesKernel<<<...>>>
    
    // 4. Download from GPU
    CopyFromGPU(particles);
    
    cudaStreamSynchronize(m_Stream);
}

void PBDFluidSolverGPU::CopyToGPU(const std::vector<FluidParticle>& particles) {
    // Need a temp buffer to convert layout?
    // FluidParticleGPU is SOA or AOS? It's defined as struct in FluidParticle.h
    std::vector<FluidParticleGPU> tempPtr(particles.size());
    for(size_t i=0; i<particles.size(); ++i) {
        tempPtr[i].FromCPU(particles[i]);
    }
    CUDA_CHECK(cudaMemcpyAsync(d_Particles, tempPtr.data(), particles.size() * sizeof(FluidParticleGPU), cudaMemcpyHostToDevice, m_Stream));
}

void PBDFluidSolverGPU::CopyFromGPU(std::vector<FluidParticle>& particles) {
    std::vector<FluidParticleGPU> tempPtr(particles.size());
    CUDA_CHECK(cudaMemcpyAsync(tempPtr.data(), d_Particles, particles.size() * sizeof(FluidParticleGPU), cudaMemcpyDeviceToHost, m_Stream));
    for(size_t i=0; i<particles.size(); ++i) {
        tempPtr[i].ToCPU(particles[i]);
    }
}

#else // !HAS_CUDA_TOOLKIT
#include "PBDFluidSolver.h"

// Stub implementation for when CUDA is disabled via CMake
PBDFluidSolverGPU::PBDFluidSolverGPU() {
    std::cerr << "Warning: PBDFluidSolverGPU instantiated but CUDA is NOT enabled. GPU acceleration will not work." << std::endl;
}

PBDFluidSolverGPU::~PBDFluidSolverGPU() {}

void PBDFluidSolverGPU::Initialize(PhysXBackend* backend) {
    // Fallback to base
    PBDFluidSolver::Initialize(backend);
}

void PBDFluidSolverGPU::Update(std::vector<FluidParticle>& particles, 
                    const std::vector<FluidType>& fluidTypes,
                    float deltaTime) {
    // Fallback to CPU
    PBDFluidSolver::Update(particles, fluidTypes, deltaTime);
}

#endif
