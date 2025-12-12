#pragma once

#include "Math/Vec3.h"

class AudioListener {
public:
    AudioListener();
    ~AudioListener();

    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }

    void MakeActive(); // Register as the global active listener

    // State sync (Called by GameObject)
    void UpdateState(const Vec3& pos, const Vec3& fwd, const Vec3& vel) {
        m_Position = pos; m_Forward = fwd; m_Velocity = vel;
    }

    const Vec3& GetPosition() const { return m_Position; }
    const Vec3& GetForward() const { return m_Forward; }
    const Vec3& GetVelocity() const { return m_Velocity; }

private:
    bool m_Enabled;
    Vec3 m_Position;
    Vec3 m_Forward;
    Vec3 m_Velocity;
};
