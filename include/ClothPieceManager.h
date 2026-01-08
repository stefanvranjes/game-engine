#pragma once

#include "ClothPiece.h"
#include "Math/Vec3.h"
#include <vector>
#include <memory>

#ifdef USE_PHYSX

class PhysXCloth;

/**
 * @brief Singleton manager for all cloth pieces in the scene
 * 
 * Handles creation, lifecycle management, and cleanup of cloth pieces
 * created from tearing operations.
 */
class ClothPieceManager {
public:
    /**
     * @brief Get singleton instance
     */
    static ClothPieceManager& GetInstance();

    /**
     * @brief Create a cloth piece from a PhysX cloth instance
     * @param cloth The cloth instance
     * @param parentID ID of the original cloth
     * @return Shared pointer to the created piece
     */
    std::shared_ptr<ClothPiece> CreatePiece(
        std::shared_ptr<PhysXCloth> cloth,
        int parentID = -1
    );

    /**
     * @brief Create two pieces from a split operation
     * @param piece1 First cloth piece
     * @param piece2 Second cloth piece
     * @param parentID ID of the original cloth
     */
    void CreatePiecesFromSplit(
        std::shared_ptr<PhysXCloth> piece1,
        std::shared_ptr<PhysXCloth> piece2,
        int parentID = -1
    );

    /**
     * @brief Update all cloth pieces
     * @param deltaTime Time since last update
     */
    void Update(float deltaTime);

    /**
     * @brief Get all active cloth pieces
     */
    const std::vector<std::shared_ptr<ClothPiece>>& GetActivePieces() const {
        return m_Pieces;
    }

    // LOD Support
    /**
     * @brief Set camera position for LOD calculations
     */
    void SetCameraPosition(const Vec3& position) { m_CameraPosition = position; }

    /**
     * @brief Update LODs for all pieces
     */
    void UpdatePieceLODs();

    /**
     * @brief Enable/disable automatic LOD for pieces
     */
    void SetLODEnabled(bool enabled) { m_LODEnabled = enabled; }

    /**
     * @brief Remove all cloth pieces
     */
    void Clear();

    /**
     * @brief Get number of active pieces
     */
    int GetPieceCount() const { return static_cast<int>(m_Pieces.size()); }

    /**
     * @brief Set maximum number of pieces allowed
     */
    void SetMaxPieces(int max) { m_MaxPieces = max; }

    /**
     * @brief Get maximum pieces limit
     */
    int GetMaxPieces() const { return m_MaxPieces; }

    /**
     * @brief Set auto-cleanup threshold (fraction of original size)
     * Pieces smaller than this threshold will be automatically removed
     */
    void SetAutoCleanupThreshold(float threshold) { 
        m_AutoCleanupThreshold = threshold; 
    }

    /**
     * @brief Get auto-cleanup threshold
     */
    float GetAutoCleanupThreshold() const { return m_AutoCleanupThreshold; }

    /**
     * @brief Set default lifetime for new pieces
     */
    void SetDefaultLifetime(float lifetime) { m_DefaultLifetime = lifetime; }

    /**
     * @brief Set default fade duration for new pieces
     */
    void SetDefaultFadeDuration(float duration) { m_DefaultFadeDuration = duration; }

private:
    ClothPieceManager();
    ~ClothPieceManager();

    // Prevent copying
    ClothPieceManager(const ClothPieceManager&) = delete;
    ClothPieceManager& operator=(const ClothPieceManager&) = delete;

    /**
     * @brief Remove destroyed pieces from the list
     */
    void CleanupDestroyedPieces();

    /**
     * @brief Enforce maximum piece limit by removing oldest/smallest pieces
     */
    void EnforcePieceLimit();

    /**
     * @brief Check and remove pieces that are too small
     */
    void CleanupSmallPieces();

    std::vector<std::shared_ptr<ClothPiece>> m_Pieces;
    
    int m_NextPieceID;
    int m_MaxPieces; // Kept as it was not explicitly removed by the instruction
    // Cleanup thresholds
    float m_AutoCleanupThreshold;
    
    // Default settings
    float m_DefaultLifetime;
    float m_DefaultFadeDuration;

    // LOD support
    Vec3 m_CameraPosition;
    bool m_LODEnabled;
};

#endif // USE_PHYSX
