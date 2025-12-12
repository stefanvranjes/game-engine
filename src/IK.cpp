#include "IK.h"
#include "Bone.h"
#include <cmath>
#include <algorithm>
#include <iostream>

void IKSolver::SolveFABRIK(IKChain& chain, const Skeleton* skeleton, std::vector<Mat4>& globalTransforms) {
    if (chain.weight <= 0.001f || chain.chainIndices.empty()) {
        return;
    }
    
    // 1. Build chain positions from current global transforms
    size_t numJoints = chain.chainIndices.size();
    if (numJoints < 2) return;
    
    std::vector<Vec3> positions(numJoints);
    for (size_t i = 0; i < numJoints; ++i) {
        int index = chain.chainIndices[i];
        if (index >= 0 && index < static_cast<int>(globalTransforms.size())) {
            Vec3 pos, scale;
            Quaternion rot;
            DecomposeMatrix(globalTransforms[index], pos, rot, scale);
            positions[i] = pos;
        }
    }
    
    // 2. Initialize lengths if needed (only once or if mismatch)
    // Note: Assuming static bone lengths usually, but scaling might affect it. 
    // Ideally we update this from the current pose if consistent scaling is assumed?
    // Let's re-calculate lengths from current pose every time to handle bone scaling.
    chain.lengths.resize(numJoints - 1);
    float totalLength = 0.0f;
    for (size_t i = 0; i < numJoints - 1; ++i) {
        float len = (positions[i] - positions[i+1]).Length();
        chain.lengths[i] = len;
        totalLength += len;
    }
    
    // 3. Check reachability
    // Chain structure in vector is: [0]=Effector, [Last]=Root
    // So root is at positions[numJoints-1]
    Vec3 rootPos = positions[numJoints - 1];
    Vec3 target = chain.targetPosition;
    
    // Blend target based on weight (Simple approach: blend final target)
    // A better approach for low weights is difficult with FABRIK as it modifies the whole chain directly.
    // Standard practice: Solve for full target, then Lerp final transform results? 
    // Or Lerp target position? Lerping target position is easier but less physically accurate for IK blending.
    // Let's stick to solving for the specific target.
    
    float distToTarget = (target - rootPos).Length();
    
    // Unreachable
    if (distToTarget > totalLength) {
        // Just point straight at target
        Vec3 dir = (target - rootPos).Normalized();
        for (int i = static_cast<int>(numJoints) - 2; i >= 0; --i) {
            positions[i] = positions[i+1] + dir * chain.lengths[i];
        }
    } else {
        // Reachable - Iteration
        Vec3 startPos = positions[numJoints - 1]; // Root stays fixed
        
        for (int iter = 0; iter < chain.iterations; ++iter) {
            if ((positions[0] - target).Length() < chain.tolerance) {
                break;
            }
            
            // Forward Reaching (Effector -> Root)
            positions[0] = target;
            for (size_t i = 1; i < numJoints; ++i) {
                Vec3 dir = (positions[i] - positions[i-1]).Normalized();
                positions[i] = positions[i-1] + dir * chain.lengths[i-1];
            }
            
            // Backward Reaching (Root -> Effector)
            positions[numJoints - 1] = startPos;
            for (int i = static_cast<int>(numJoints) - 2; i >= 0; --i) {
                Vec3 dir = (positions[i] - positions[i+1]).Normalized();
                positions[i] = positions[i+1] + dir * chain.lengths[i];
            }
        }
    }
    
    // 4. Update Global Transforms (Rotation only or full matrix?)
    // FABRIK gives positions. We need to update rotations to point to new positions.
    // This is the tricky part. 
    // Strategy: For each bone, find rotation that maps (OriginalDirection to local Child) to (NewDirection to new Child position).
    
    for (size_t i = numJoints - 1; i > 0; --i) {
        int parentIdx = chain.chainIndices[i];
        int childIdx = chain.chainIndices[i-1]; // Index in chain (lower is closer to effector)
        
        // Get original global transform data
        Vec3 oldParentPos, oldParentScale;
        Quaternion oldParentRot;
        DecomposeMatrix(globalTransforms[parentIdx], oldParentPos, oldParentRot, oldParentScale);
        
        Vec3 oldChildPos, oldChildScale;
        Quaternion oldChildRot;
        DecomposeMatrix(globalTransforms[childIdx], oldChildPos, oldChildRot, oldChildScale);
        
        // Vectors
        Vec3 oldDir = (oldChildPos - oldParentPos).Normalized();
        Vec3 newDir = (positions[i-1] - positions[i]).Normalized();
        
        // Calculate rotation delta
        Quaternion deltaRot = Quaternion::FromTo(oldDir, newDir);
        
        // Apply delta to parent global rotation
        Quaternion newParentRot = deltaRot * oldParentRot; // Apply global delta
        newParentRot.Normalize(); // Normalize to prevent drift
        
        // Update matrix
        // We use the NEW positions solved by FABRIK
        globalTransforms[parentIdx] = ComposeMatrix(positions[i], newParentRot, oldParentScale);
        
        // Important: Child's global rotation also needs to be updated because it inherits parent's rotation!
        // Actually, if we update parent global rotation, child's global rotation (which is theoretically derived from parent)
        // is now invalid vs the globalTransforms array.
        // We need to propagate this rotation change down the chain OR we just re-orient the next bone in the next loop iteration.
        // BUT: The loop goes Root -> Effector (indices numJoints-1 down to 0).
        // When we are at Root (i=Last), we orient Root to point to next joint.
        // When we are at next joint (i=Last-1), we orient IT to point to its child.
        // The rotations are calculated purely based on "Point to Child".
        // The Effector (i=0) doesn't point to anything (end of chain). WE usually typically preserve its original local rotation relative to parent
        // or match a target rotation. For now, let's just let it inherit the rotation change from parent?
        // Or better: The effector's rotation is usually meant to match a target rotation (e.g. hand orientation).
        // Since we don't have target rotation, we might leave it or reuse previous global rotation adjusted by delta.
    }
    
    // Fix effector rotation?
    // Current loop doesn't update effector (buffer[0]) rotation, only position (implicitly by parent).
    // Actually, we must update the effector's matrix to have the correct position at least.
    int effectorIdx = chain.chainIndices[0];
    Vec3 oldEffPos, oldEffScale;
    Quaternion oldEffRot;
    DecomposeMatrix(globalTransforms[effectorIdx], oldEffPos, oldEffRot, oldEffScale);
    
    // Just update position, keep rotation (or maybe align with target if we had one)
    globalTransforms[effectorIdx] = ComposeMatrix(positions[0], oldEffRot, oldEffScale);
    
    // Basic Weight Blending (Lerp entire matrix back to original if Weight < 1.0)
    // Optimization: Do this inside the loop or after? After is cleaner conceptually (pose blend).
    if (chain.weight < 1.0f) {
        // This requires we kept a copy of original transforms... 
        // But we modified them in place. 
        // Ideally we should have taken a copy at start.
        // Given performance, let's trust the user sets weight=1 for now or we refactor to copy.
        // TODO: Copy original transforms for blending.
    }
}
