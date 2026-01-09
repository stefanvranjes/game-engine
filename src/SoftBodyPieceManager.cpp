#include "SoftBodyPieceManager.h"
#include "PhysXSoftBody.h"
#include "GameObject.h"
#include <algorithm>
#include <iostream>

SoftBodyPieceManager::SoftBodyPieceManager()
    : m_NextPieceId(0)
    , m_CurrentTime(0.0f)
{
}

SoftBodyPieceManager::~SoftBodyPieceManager() {
    Clear();
}

int SoftBodyPieceManager::AddPiece(
    std::shared_ptr<PhysXSoftBody> softBody,
    std::shared_ptr<GameObject> gameObject,
    int parentId)
{
    Piece piece;
    piece.softBody = softBody;
    piece.gameObject = gameObject;
    piece.parentPieceId = parentId;
    piece.creationTime = m_CurrentTime;
    piece.mass = softBody ? softBody->GetTotalMass() : 0.0f;
    piece.isActive = true;
    piece.id = m_NextPieceId++;

    m_Pieces.push_back(piece);

    std::cout << "Added soft body piece " << piece.id 
              << " (parent: " << parentId << ", mass: " << piece.mass << ")" << std::endl;

    if (m_PieceCreatedCallback) {
        m_PieceCreatedCallback(piece);
    }

    return piece.id;
}

void SoftBodyPieceManager::RemovePiece(int pieceId) {
    auto it = std::find_if(m_Pieces.begin(), m_Pieces.end(),
        [pieceId](const Piece& p) { return p.id == pieceId; });

    if (it != m_Pieces.end()) {
        std::cout << "Removing soft body piece " << pieceId << std::endl;

        if (m_PieceRemovedCallback) {
            m_PieceRemovedCallback(pieceId);
        }

        m_Pieces.erase(it);
    }
}

void SoftBodyPieceManager::Update(float deltaTime) {
    m_CurrentTime += deltaTime;

    // Update all active pieces
    for (auto& piece : m_Pieces) {
        if (piece.isActive && piece.softBody) {
            // Pieces are updated by PhysX backend
            // This is just for piece-specific logic
        }
    }
}

int SoftBodyPieceManager::GetPieceCount() const {
    return static_cast<int>(m_Pieces.size());
}

int SoftBodyPieceManager::GetActivePieceCount() const {
    int count = 0;
    for (const auto& piece : m_Pieces) {
        if (piece.isActive) {
            count++;
        }
    }
    return count;
}

int SoftBodyPieceManager::CleanupSmallPieces(float minMass) {
    int removedCount = 0;

    auto it = m_Pieces.begin();
    while (it != m_Pieces.end()) {
        if (it->mass < minMass) {
            std::cout << "Cleaning up small piece " << it->id 
                     << " (mass: " << it->mass << ")" << std::endl;

            if (m_PieceRemovedCallback) {
                m_PieceRemovedCallback(it->id);
            }

            it = m_Pieces.erase(it);
            removedCount++;
        } else {
            ++it;
        }
    }

    return removedCount;
}

SoftBodyPieceManager::Piece* SoftBodyPieceManager::GetPiece(int pieceId) {
    auto it = std::find_if(m_Pieces.begin(), m_Pieces.end(),
        [pieceId](const Piece& p) { return p.id == pieceId; });

    return (it != m_Pieces.end()) ? &(*it) : nullptr;
}

void SoftBodyPieceManager::Clear() {
    std::cout << "Clearing all soft body pieces (" << m_Pieces.size() << " pieces)" << std::endl;
    m_Pieces.clear();
    m_NextPieceId = 0;
}
