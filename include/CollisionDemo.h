#pragma once

#include "ParticleSystem.h"
#include <string>

// Visual demonstration class for particle collision system
// Cycles through 5 different demo scenes showcasing collision features
class CollisionDemo {
public:
    CollisionDemo(ParticleSystem* particleSystem);

    void Init();
    void Update(float deltaTime);
    void NextDemo();
    std::string GetCurrentDemoName() const;

private:
    void LoadDemo1_GroundBounce();
    void LoadDemo2_SphereObstacle();
    void LoadDemo3_BoxContainer();
    void LoadDemo4_ParticleCollisions();
    void LoadDemo5_ComplexScene();

    ParticleSystem* m_ParticleSystem;
    int m_CurrentDemo;
    float m_DemoTimer;
};
