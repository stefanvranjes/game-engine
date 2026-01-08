#pragma once

#include "IPhysicsCloth.h"
#include "Math/Vec3.h"
#include <memory>
#include <string>

#ifdef USE_PHYSX

class PhysXCloth;

/**
 * @brief Represents a single cloth piece with lifecycle management
 * 
 * Wraps a PhysXCloth instance and manages its lifecycle state,
 * fade-out animation, and cleanup logic.
 */
class ClothPiece {
public:
    enum class State {
        Active,      // Normal simulation
        Fading,      // Fading out before destruction
        Destroyed    // Ready for removal
    };

    /**
     * @brief Construct a cloth piece from a PhysX cloth instance
     * @param cloth The PhysX cloth instance
     * @param parentID ID of the original cloth this piece came from
     * @param pieceID Unique ID for this piece
     */
    ClothPiece(
        std::shared_ptr<PhysXCloth> cloth,
        int parentID = -1,
        int pieceID = -1
    );

    ~ClothPiece();

    /**
     * @brief Update the cloth piece
     * @param deltaTime Time since last update
     */
    void Update(float deltaTime);

    /**
     * @brief Get the underlying PhysX cloth instance
     */
    std::shared_ptr<PhysXCloth> GetCloth() const { return m_Cloth; }

    /**
     * @brief Get current lifecycle state
     */
    State GetState() const { return m_State; }

    /**
     * @brief Get current alpha/opacity (for fade-out)
     * @return Alpha value from 0.0 (transparent) to 1.0 (opaque)
     */
    float GetAlpha() const { return m_Alpha; }

    /**
     * @brief Check if piece should be destroyed
     */
    bool ShouldDestroy() const { return m_State == State::Destroyed; }

    /**
     * @brief Get approximate size of the piece
     * @return Size metric (particle count * average edge length)
     */
    float GetSize() const;

    /**
     * @brief Get piece ID
     */
    int GetPieceID() const { return m_PieceID; }

    /**
     * @brief Get parent cloth ID
     */
    int GetParentID() const { return m_ParentID; }

    /**
     * @brief Get age of the piece in seconds
     */
    float GetAge() const { return m_Age; }

    /**
     * @brief Trigger fade-out and destruction
     */
    void StartFadeOut();

    /**
     * @brief Set maximum lifetime before auto fade-out
     */
    void SetLifetime(float lifetime) { m_MaxLifetime = lifetime; }

    /**
     * @brief Set fade-out duration
     */
    void SetFadeDuration(float duration) { m_FadeDuration = duration; }

private:
    std::shared_ptr<PhysXCloth> m_Cloth;
    
    int m_PieceID;
    int m_ParentID;
    
    State m_State;
    float m_Age;
    float m_Alpha;
    
    float m_MaxLifetime;    // Auto fade-out after this time
    float m_FadeDuration;   // Duration of fade-out animation
    float m_FadeTimer;      // Current fade progress
    
    // Cached size data
    mutable float m_CachedSize;
    mutable bool m_SizeCached;
};

#endif // USE_PHYSX
