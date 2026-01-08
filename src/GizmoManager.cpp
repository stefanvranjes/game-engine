#include "GizmoManager.h"
#include "GameObject.h"
#include "Camera.h"
#include <GLFW/glfw3.h>

GizmoManager::GizmoManager() {
    m_TranslationGizmo = std::make_unique<TranslationGizmo>();
    m_RotationGizmo = std::make_unique<RotationGizmo>();
    m_ScaleGizmo = std::make_unique<ScaleGizmo>();
}

void GizmoManager::SetSelectedObject(std::shared_ptr<GameObject> object) {
    m_SelectedObject = object;
    
    if (m_SelectedObject) {
         m_TranslationGizmo->SetTransform(&m_SelectedObject->GetTransform());
         m_RotationGizmo->SetTransform(&m_SelectedObject->GetTransform());
         m_ScaleGizmo->SetTransform(&m_SelectedObject->GetTransform());
    } else {
         m_TranslationGizmo->SetTransform(nullptr);
         m_RotationGizmo->SetTransform(nullptr);
         m_ScaleGizmo->SetTransform(nullptr);
    }
}

void GizmoManager::SetGizmoType(GizmoType type) {
    m_CurrentType = type;
}

Gizmo* GizmoManager::GetActiveGizmo() {
    switch (m_CurrentType) {
        case GizmoType::Translation: return m_TranslationGizmo.get();
        case GizmoType::Rotation: return m_RotationGizmo.get();
        case GizmoType::Scale: return m_ScaleGizmo.get();
        default: return nullptr;
    }
}

bool GizmoManager::IsDragging() const {
    if (m_CurrentType == GizmoType::Translation) return m_TranslationGizmo->IsDragging();
    if (m_CurrentType == GizmoType::Rotation) return m_RotationGizmo->IsDragging();
    if (m_CurrentType == GizmoType::Scale) return m_ScaleGizmo->IsDragging();
    return false;
}

void GizmoManager::Update(float deltaTime) {
    // Propagate snap settings to all gizmos (or just active one)
    m_TranslationGizmo->SetSnapping(m_SnappingEnabled, m_TranslationSnap);
    m_RotationGizmo->SetSnapping(m_SnappingEnabled, m_RotationSnap);
    m_ScaleGizmo->SetSnapping(m_SnappingEnabled, m_ScaleSnap);
}

void GizmoManager::Render(Shader* shader, const Camera& camera) {
    if (!m_SelectedObject) return;
    
    Gizmo* active = GetActiveGizmo();
    if (active) {
        // Ensure shader is set up for unlit/flat color rendering
        // Assuming 'shader' passed here is a simple unlit shader
        active->Draw(shader, camera);
    }
}

void GizmoManager::ProcessInput(GLFWwindow* window, float deltaTime, const Camera& camera) {
    if (!m_SelectedObject) return;
    
    // Keyboard Shortcuts
    static bool wPressed = false;
    static bool ePressed = false;
    static bool rPressed = false;
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        if (!wPressed) SetGizmoType(GizmoType::Translation);
        wPressed = true;
    } else wPressed = false;
    
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        if (!ePressed) SetGizmoType(GizmoType::Rotation);
        ePressed = true;
    } else ePressed = false;
    
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        if (!rPressed) SetGizmoType(GizmoType::Scale);
        rPressed = true;
    } else rPressed = false;
    
    // Mouse Input
    Gizmo* active = GetActiveGizmo();
    if (!active) return;
    
    // Get Mouse Position and create Ray
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    
    // Normalized Device Coordinates
    float ndcX = (2.0f * (float)xpos) / width - 1.0f;
    float ndcY = 1.0f - (2.0f * (float)ypos) / height;
    
    // Create Ray
    Mat4 invView = camera.GetViewMatrix().Inverse();
    Mat4 invProj = camera.GetProjectionMatrix().Inverse();
    
    Vec4 rayClip(ndcX, ndcY, -1.0f, 1.0f);
    Vec4 rayEye = invProj * rayClip;
    rayEye = Vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    
    Vec4 rayWorld4 = invView * rayEye;
    Vec3 rayWorld(rayWorld4.x, rayWorld4.y, rayWorld4.z);
    rayWorld = rayWorld.Normalized();
    
    Ray ray;
    ray.origin = camera.GetPosition();
    ray.direction = rayWorld;
    
    // Interaction
    bool mouseLeft = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    
    if (mouseLeft && !m_MousePressed) {
        // Press event
        if (active->OnMousePress(ray)) {
            // Gizmo handled it
        }
        m_MousePressed = true;
    } 
    else if (!mouseLeft && m_MousePressed) {
        // Release event
        active->OnMouseRelease();
        m_MousePressed = false;
    }
    else if (mouseLeft && m_MousePressed) {
        // Drag event
        if (active->IsDragging()) {
            active->OnMouseDrag(ray, camera);
        }
    }
    else {
        // Move/Hover
        active->OnMouseMove(ray);
    }
}
