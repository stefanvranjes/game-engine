#include "ClothPiece.h"

#ifdef USE_PHYSX

#include "PhysXCloth.h"
#include <iostream>

ClothPiece::ClothPiece(
    std::shared_ptr<PhysXCloth> cloth,
    int parentID,
    int pieceID)
    : m_Cloth(cloth)
    , m_PieceID(pieceID)
    , m_ParentID(parentID)
    , m_State(State::Active)
    , m_Age(0.0f)
    , m_Alpha(1.0f)
    , m_MaxLifetime(30.0f)      // 30 seconds default
    , m_FadeDuration(2.0f)      // 2 second fade
    , m_FadeTimer(0.0f)
    , m_CachedSize(-1.0f)
    , m_SizeCached(false)
{
    std::cout << "ClothPiece created: ID=" << m_PieceID 
              << ", Parent=" << m_ParentID << std::endl;
}

ClothPiece::~ClothPiece() {
    std::cout << "ClothPiece destroyed: ID=" << m_PieceID << std::endl;
}

void ClothPiece::Update(float deltaTime) {
    if (!m_Cloth) {
        m_State = State::Destroyed;
        return;
    }

    m_Age += deltaTime;

    switch (m_State) {
        case State::Active:
            // Update physics simulation
            m_Cloth->Update(deltaTime);

            // Check if lifetime expired
            if (m_MaxLifetime > 0.0f && m_Age >= m_MaxLifetime) {
                StartFadeOut();
            }
            break;

        case State::Fading:
            // Continue physics update during fade
            m_Cloth->Update(deltaTime);

            // Update fade animation
            m_FadeTimer += deltaTime;
            m_Alpha = 1.0f - (m_FadeTimer / m_FadeDuration);

            if (m_Alpha <= 0.0f) {
                m_Alpha = 0.0f;
                m_State = State::Destroyed;
                std::cout << "ClothPiece faded out: ID=" << m_PieceID << std::endl;
            }
            break;

        case State::Destroyed:
            // Do nothing, waiting for removal
            break;
    }
}

float ClothPiece::GetSize() const {
    if (m_SizeCached) {
        return m_CachedSize;
    }

    if (!m_Cloth) {
        m_CachedSize = 0.0f;
        m_SizeCached = true;
        return 0.0f;
    }

    // Calculate size based on particle count and bounding box
    int particleCount = m_Cloth->GetParticleCount();
    
    if (particleCount == 0) {
        m_CachedSize = 0.0f;
        m_SizeCached = true;
        return 0.0f;
    }

    // Get particle positions to calculate bounding box
    std::vector<Vec3> positions(particleCount);
    m_Cloth->GetParticlePositions(positions.data());

    // Calculate bounding box
    Vec3 minBounds = positions[0];
    Vec3 maxBounds = positions[0];

    for (int i = 1; i < particleCount; ++i) {
        minBounds.x = std::min(minBounds.x, positions[i].x);
        minBounds.y = std::min(minBounds.y, positions[i].y);
        minBounds.z = std::min(minBounds.z, positions[i].z);

        maxBounds.x = std::max(maxBounds.x, positions[i].x);
        maxBounds.y = std::max(maxBounds.y, positions[i].y);
        maxBounds.z = std::max(maxBounds.z, positions[i].z);
    }

    Vec3 size = maxBounds - minBounds;
    float volume = size.x * size.y * size.z;

    // Size metric: volume * particle count
    m_CachedSize = volume * static_cast<float>(particleCount);
    m_SizeCached = true;

    return m_CachedSize;
}

void ClothPiece::StartFadeOut() {
    if (m_State == State::Active) {
        m_State = State::Fading;
        m_FadeTimer = 0.0f;
        std::cout << "ClothPiece starting fade-out: ID=" << m_PieceID << std::endl;
    }
}

#endif // USE_PHYSX
