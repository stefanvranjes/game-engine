#pragma once

#include "IPhysicsSoftBody.h"
#include <vector>
#include <memory>
#include <functional>

class PhysXBackend;
class PhysXSoftBody;
class GameObject;

/**
 * @brief Manages torn soft body pieces
 */
class SoftBodyPieceManager {
public:
    /**
     * @brief Information about a soft body piece
     */
    struct Piece {
        std::shared_ptr<PhysXSoftBody> softBody;
        std::shared_ptr<GameObject> gameObject;
        int parentPieceId;
        float creationTime;
        float mass;
        bool isActive;
        int id;
    };

    SoftBodyPieceManager();
    ~SoftBodyPieceManager();

    /**
     * @brief Add a new piece
     * 
     * @param softBody Soft body physics component
     * @param gameObject Game object containing the piece
     * @param parentId ID of parent piece (-1 if original)
     * @return ID of the new piece
     */
    int AddPiece(
        std::shared_ptr<PhysXSoftBody> softBody,
        std::shared_ptr<GameObject> gameObject,
        int parentId = -1
    );

    /**
     * @brief Remove a piece
     */
    void RemovePiece(int pieceId);

    /**
     * @brief Update all pieces
     */
    void Update(float deltaTime);

    /**
     * @brief Get piece count
     */
    int GetPieceCount() const;

    /**
     * @brief Get active piece count
     */
    int GetActivePieceCount() const;

    /**
     * @brief Cleanup small pieces below mass threshold
     * 
     * @param minMass Minimum mass to keep
     * @return Number of pieces removed
     */
    int CleanupSmallPieces(float minMass);

    /**
     * @brief Get piece by ID
     */
    Piece* GetPiece(int pieceId);

    /**
     * @brief Get all pieces
     */
    const std::vector<Piece>& GetAllPieces() const { return m_Pieces; }

    /**
     * @brief Set callback for piece creation
     */
    void SetPieceCreatedCallback(std::function<void(const Piece&)> callback) {
        m_PieceCreatedCallback = callback;
    }

    /**
     * @brief Set callback for piece removal
     */
    void SetPieceRemovedCallback(std::function<void(int)> callback) {
        m_PieceRemovedCallback = callback;
    }

    /**
     * @brief Clear all pieces
     */
    void Clear();

private:
    std::vector<Piece> m_Pieces;
    int m_NextPieceId;
    float m_CurrentTime;

    std::function<void(const Piece&)> m_PieceCreatedCallback;
    std::function<void(int)> m_PieceRemovedCallback;
};
