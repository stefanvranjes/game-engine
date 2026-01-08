#include "ClothPieceManager.h"

#ifdef USE_PHYSX

#include "PhysXCloth.h"
#include <algorithm>
#include <iostream>

ClothPieceManager::ClothPieceManager()
    : m_NextPieceID(0)
    , m_MaxPieces(20)
    , m_AutoCleanupThreshold(0.05f)  // 5% of original size
    , m_DefaultLifetime(30.0f)        // 30 seconds
    , m_DefaultFadeDuration(2.0f)     // 2 seconds
    , m_CameraPosition(0, 0, 0)
    , m_LODEnabled(true)
{
    std::cout << "ClothPieceManager initialized" << std::endl;
}

ClothPieceManager::~ClothPieceManager() {
    Clear();
    std::cout << "ClothPieceManager destroyed" << std::endl;
}

ClothPieceManager& ClothPieceManager::GetInstance() {
    static ClothPieceManager instance;
    return instance;
}

std::shared_ptr<ClothPiece> ClothPieceManager::CreatePiece(
    std::shared_ptr<PhysXCloth> cloth,
    int parentID)
{
    if (!cloth) {
        std::cerr << "Cannot create piece from null cloth" << std::endl;
        return nullptr;
    }

    // Create new piece
    auto piece = std::make_shared<ClothPiece>(cloth, parentID, m_NextPieceID++);
    piece->SetLifetime(m_DefaultLifetime);
    piece->SetFadeDuration(m_DefaultFadeDuration);

    m_Pieces.push_back(piece);

    std::cout << "Created cloth piece " << piece->GetPieceID() 
              << " (Total: " << m_Pieces.size() << ")" << std::endl;

    // Enforce piece limit
    EnforcePieceLimit();

    return piece;
}

void ClothPieceManager::CreatePiecesFromSplit(
    std::shared_ptr<PhysXCloth> piece1,
    std::shared_ptr<PhysXCloth> piece2,
    int parentID)
{
    if (!piece1 || !piece2) {
        std::cerr << "Cannot create pieces from null cloth" << std::endl;
        return;
    }

    CreatePiece(piece1, parentID);
    CreatePiece(piece2, parentID);

    std::cout << "Created 2 pieces from split (Parent: " << parentID << ")" << std::endl;
}

void ClothPieceManager::Update(float deltaTime) {
    // Update all pieces
    for (auto& piece : m_Pieces) {
        piece->Update(deltaTime);
    }

    // Cleanup
    CleanupDestroyedPieces();
    CleanupSmallPieces();
}

void ClothPieceManager::Clear() {
    m_Pieces.clear();
    std::cout << "Cleared all cloth pieces" << std::endl;
}

void ClothPieceManager::CleanupDestroyedPieces() {
    auto it = std::remove_if(m_Pieces.begin(), m_Pieces.end(),
        [](const std::shared_ptr<ClothPiece>& piece) {
            return piece->ShouldDestroy();
        });

    if (it != m_Pieces.end()) {
        int removedCount = static_cast<int>(std::distance(it, m_Pieces.end()));
        m_Pieces.erase(it, m_Pieces.end());
        std::cout << "Removed " << removedCount << " destroyed pieces" << std::endl;
    }
}

void ClothPieceManager::EnforcePieceLimit() {
    if (static_cast<int>(m_Pieces.size()) <= m_MaxPieces) {
        return;
    }

    int toRemove = static_cast<int>(m_Pieces.size()) - m_MaxPieces;
    
    std::cout << "Piece limit exceeded (" << m_Pieces.size() 
              << "/" << m_MaxPieces << "), removing " << toRemove << " pieces" << std::endl;

    // Sort by age (oldest first) and size (smallest first)
    std::sort(m_Pieces.begin(), m_Pieces.end(),
        [](const std::shared_ptr<ClothPiece>& a, const std::shared_ptr<ClothPiece>& b) {
            // Prioritize removing older pieces
            if (std::abs(a->GetAge() - b->GetAge()) > 1.0f) {
                return a->GetAge() > b->GetAge();
            }
            // If similar age, remove smaller pieces
            return a->GetSize() < b->GetSize();
        });

    // Start fade-out for oldest/smallest pieces
    for (int i = 0; i < toRemove && i < static_cast<int>(m_Pieces.size()); ++i) {
        m_Pieces[i]->StartFadeOut();
    }
}

void ClothPieceManager::CleanupSmallPieces() {
    if (m_AutoCleanupThreshold <= 0.0f) {
        return;
    }

    // Find the largest piece to use as reference
    float maxSize = 0.0f;
    for (const auto& piece : m_Pieces) {
        float size = piece->GetSize();
        if (size > maxSize) {
            maxSize = size;
        }
    }

    if (maxSize <= 0.0f) {
        return;
    }

    // Calculate threshold size
    float thresholdSize = maxSize * m_AutoCleanupThreshold;

    // Start fade-out for pieces below threshold
    int fadedCount = 0;
    for (auto& piece : m_Pieces) {
        if (piece->GetState() == ClothPiece::State::Active) {
            if (piece->GetSize() < thresholdSize) {
                piece->StartFadeOut();
                fadedCount++;
            }
        }
    }

    if (fadedCount > 0) {
        std::cout << "Started fade-out for " << fadedCount 
                  << " small pieces (threshold: " << thresholdSize << ")" << std::endl;
    }
}

void ClothPieceManager::UpdatePieceLODs() {
    if (!m_LODEnabled) {
        return;
    }
    
    // Update LOD for each piece based on distance from camera
    for (auto& piece : m_Pieces) {
        auto cloth = piece->GetCloth();
        if (!cloth) {
            continue;
        }
        
        // Get cloth position (approximate from first particle)
        const auto& positions = cloth->GetParticlePositions();
        if (positions.empty()) {
            continue;
        }
        
        Vec3 clothPos = positions[0];  // Use first particle as reference
        
        // Calculate distance to camera
        Vec3 delta = clothPos - m_CameraPosition;
        float distance = std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
        
        // Get appropriate LOD for distance
        const ClothLODConfig& config = cloth->GetLODConfig();
        int targetLOD = config.GetLODForDistance(distance, cloth->GetCurrentLOD());
        
        // Apply LOD if changed
        if (targetLOD != cloth->GetCurrentLOD()) {
            cloth->SetLOD(targetLOD);
        }
    }
}

#endif // USE_PHYSX
