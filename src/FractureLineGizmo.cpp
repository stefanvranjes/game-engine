#include "FractureLineGizmo.h"
#include "PhysXSoftBody.h"
#include "FractureLineToPattern.h"
#include "FractureLinePatternLibrary.h"
#include "Camera.h"
#include "Shader.h"
#include <GL/glew.h>
#include <algorithm>
#include <cmath>

// Static member initialization
unsigned int FractureLineGizmo::s_SphereVAO = 0;
unsigned int FractureLineGizmo::s_SphereVBO = 0;
unsigned int FractureLineGizmo::s_SphereEBO = 0;
unsigned int FractureLineGizmo::s_CylinderVAO = 0;
unsigned int FractureLineGizmo::s_CylinderVBO = 0;
unsigned int FractureLineGizmo::s_CylinderEBO = 0;
bool FractureLineGizmo::s_GeometryInitialized = false;

FractureLineGizmo::FractureLineGizmo() {
    m_Type = GizmoType::None;
    InitGeometry();
}

void FractureLineGizmo::InitGeometry() {
    if (s_GeometryInitialized) return;
    
    // Create sphere geometry for control points
    const int segments = 16;
    const int rings = 8;
    std::vector<float> sphereVertices;
    std::vector<unsigned int> sphereIndices;
    
    for (int ring = 0; ring <= rings; ++ring) {
        float phi = 3.14159f * float(ring) / float(rings);
        for (int seg = 0; seg <= segments; ++seg) {
            float theta = 2.0f * 3.14159f * float(seg) / float(segments);
            
            float x = std::sin(phi) * std::cos(theta);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(theta);
            
            sphereVertices.push_back(x);
            sphereVertices.push_back(y);
            sphereVertices.push_back(z);
        }
    }
    
    for (int ring = 0; ring < rings; ++ring) {
        for (int seg = 0; seg < segments; ++seg) {
            int current = ring * (segments + 1) + seg;
            int next = current + segments + 1;
            
            sphereIndices.push_back(current);
            sphereIndices.push_back(next);
            sphereIndices.push_back(current + 1);
            
            sphereIndices.push_back(current + 1);
            sphereIndices.push_back(next);
            sphereIndices.push_back(next + 1);
        }
    }
    
    glGenVertexArrays(1, &s_SphereVAO);
    glGenBuffers(1, &s_SphereVBO);
    glGenBuffers(1, &s_SphereEBO);
    
    glBindVertexArray(s_SphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_SphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), sphereVertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_SphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(unsigned int), sphereIndices.data(), GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    glBindVertexArray(0);
    
    // Create cylinder geometry for line segments
    const int cylinderSegments = 12;
    std::vector<float> cylinderVertices;
    std::vector<unsigned int> cylinderIndices;
    
    for (int i = 0; i <= cylinderSegments; ++i) {
        float theta = 2.0f * 3.14159f * float(i) / float(cylinderSegments);
        float x = std::cos(theta);
        float z = std::sin(theta);
        
        // Bottom circle
        cylinderVertices.push_back(x);
        cylinderVertices.push_back(0.0f);
        cylinderVertices.push_back(z);
        
        // Top circle
        cylinderVertices.push_back(x);
        cylinderVertices.push_back(1.0f);
        cylinderVertices.push_back(z);
    }
    
    for (int i = 0; i < cylinderSegments; ++i) {
        int bottom1 = i * 2;
        int top1 = i * 2 + 1;
        int bottom2 = (i + 1) * 2;
        int top2 = (i + 1) * 2 + 1;
        
        cylinderIndices.push_back(bottom1);
        cylinderIndices.push_back(bottom2);
        cylinderIndices.push_back(top1);
        
        cylinderIndices.push_back(top1);
        cylinderIndices.push_back(bottom2);
        cylinderIndices.push_back(top2);
    }
    
    glGenVertexArrays(1, &s_CylinderVAO);
    glGenBuffers(1, &s_CylinderVBO);
    glGenBuffers(1, &s_CylinderEBO);
    
    glBindVertexArray(s_CylinderVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_CylinderVBO);
    glBufferData(GL_ARRAY_BUFFER, cylinderVertices.size() * sizeof(float), cylinderVertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_CylinderEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cylinderIndices.size() * sizeof(unsigned int), cylinderIndices.data(), GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    glBindVertexArray(0);
    
    s_GeometryInitialized = true;
}

void FractureLineGizmo::Update(float deltaTime) {
    // Update logic if needed
}

void FractureLineGizmo::Draw(Shader* shader, const Camera& camera) {
    if (!m_Enabled || !m_SoftBody) return;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    
    for (int i = 0; i < static_cast<int>(fractureLines.size()); ++i) {
        DrawFractureLine(shader, camera, fractureLines[i], i);
    }
}

void FractureLineGizmo::DrawFractureLine(Shader* shader, const Camera& camera, const FractureLine& line, int lineIndex) {
    const auto& points = line.GetPoints();
    if (points.empty()) return;
    
    bool isSelected = (lineIndex == m_SelectedLineIndex);
    Vec3 lineColor = isSelected ? m_ColorSelected : GetColorForWeakness(line.GetWeaknessMultiplier());
    
    // Draw line segments
    for (size_t i = 0; i < points.size() - 1; ++i) {
        bool isHoveredSegment = (lineIndex == m_SelectedLineIndex && static_cast<int>(i) == m_HoveredSegmentIndex);
        Vec3 segmentColor = isHoveredSegment ? m_ColorHovered : lineColor;
        DrawLineSegment(shader, points[i], points[i + 1], segmentColor, m_LineThickness);
    }
    
    // Draw width visualization if enabled
    if (m_ShowWidthVisualization && isSelected) {
        DrawWidthVisualization(shader, camera, line);
    }
    
    // Draw control points
    for (size_t i = 0; i < points.size(); ++i) {
        bool isHovered = (lineIndex == m_SelectedLineIndex && static_cast<int>(i) == m_HoveredPointIndex);
        bool isSelectedPoint = (lineIndex == m_SelectedLineIndex && static_cast<int>(i) == m_SelectedPointIndex);
        
        Vec3 pointColor = isSelectedPoint ? m_ColorSelected : (isHovered ? m_ColorHovered : lineColor);
        float pointSize = (isHovered || isSelectedPoint) ? m_PointSize * 1.5f : m_PointSize;
        
        DrawControlPoint(shader, points[i], pointColor, pointSize);
    }
}

void FractureLineGizmo::DrawControlPoint(Shader* shader, const Vec3& position, const Vec3& color, float size) {
    Mat4 model;
    model.SetIdentity();
    model.SetTranslation(position);
    model.Scale(size, size, size);
    
    shader->SetMat4("model", model.m);
    shader->SetVec3("color", color.x, color.y, color.z);
    
    glBindVertexArray(s_SphereVAO);
    glDrawElements(GL_TRIANGLES, 16 * 8 * 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void FractureLineGizmo::DrawLineSegment(Shader* shader, const Vec3& start, const Vec3& end, const Vec3& color, float thickness) {
    Vec3 direction = end - start;
    float length = direction.Length();
    if (length < 0.0001f) return;
    
    direction = direction * (1.0f / length);
    
    // Create transformation matrix
    Mat4 model;
    model.SetIdentity();
    model.SetTranslation(start);
    
    // Align cylinder with direction
    Vec3 up = Vec3(0, 1, 0);
    if (std::abs(direction.Dot(up)) > 0.99f) {
        up = Vec3(1, 0, 0);
    }
    
    Vec3 right = direction.Cross(up);
    right = right * (1.0f / right.Length());
    up = right.Cross(direction);
    
    // Build rotation matrix
    Mat4 rotation;
    rotation.m[0] = right.x;    rotation.m[4] = direction.x; rotation.m[8] = up.x;     rotation.m[12] = 0;
    rotation.m[1] = right.y;    rotation.m[5] = direction.y; rotation.m[9] = up.y;     rotation.m[13] = 0;
    rotation.m[2] = right.z;    rotation.m[6] = direction.z; rotation.m[10] = up.z;    rotation.m[14] = 0;
    rotation.m[3] = 0;          rotation.m[7] = 0;           rotation.m[11] = 0;       rotation.m[15] = 1;
    
    model = model * rotation;
    model.Scale(thickness, length, thickness);
    
    shader->SetMat4("model", model.m);
    shader->SetVec3("color", color.x, color.y, color.z);
    
    glBindVertexArray(s_CylinderVAO);
    glDrawElements(GL_TRIANGLES, 12 * 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void FractureLineGizmo::DrawWidthVisualization(Shader* shader, const Camera& camera, const FractureLine& line) {
    const auto& points = line.GetPoints();
    if (points.size() < 2) return;
    
    float width = line.GetWidth();
    Vec3 widthColor = m_ColorNormal;
    widthColor.x *= 0.5f;
    widthColor.y *= 0.5f;
    widthColor.z *= 0.5f;
    
    // Draw semi-transparent tubes around each segment
    for (size_t i = 0; i < points.size() - 1; ++i) {
        DrawLineSegment(shader, points[i], points[i + 1], widthColor, width * 0.5f);
    }
}

bool FractureLineGizmo::OnMousePress(const Ray& ray) {
    if (!m_Enabled || !m_SoftBody) return false;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    
    // Try to pick a point first
    float closestDistance = std::numeric_limits<float>::max();
    int closestLineIndex = -1;
    int closestPointIndex = -1;
    
    for (int i = 0; i < static_cast<int>(fractureLines.size()); ++i) {
        int pointIndex;
        float distance;
        if (PickPoint(ray, i, pointIndex, distance)) {
            if (distance < closestDistance) {
                closestDistance = distance;
                closestLineIndex = i;
                closestPointIndex = pointIndex;
            }
        }
    }
    
    if (closestLineIndex >= 0) {
        m_SelectedLineIndex = closestLineIndex;
        m_SelectedPointIndex = closestPointIndex;
        
        if (m_EditMode == EditMode::Delete) {
            RemovePoint(closestPointIndex);
            return true;
        } else if (m_EditMode == EditMode::Edit) {
            // Start dragging
            m_IsDragging = true;
            const auto& points = fractureLines[closestLineIndex].GetPoints();
            m_OriginalPointPosition = points[closestPointIndex];
            
            // Create drag plane perpendicular to camera view
            Vec3 cameraPos = Vec3(0, 0, 0); // TODO: Get from camera
            m_DragPlaneNormal = Vec3(0, 0, 1); // TODO: Calculate from camera
            m_DragStartPoint = m_OriginalPointPosition;
            
            return true;
        }
    }
    
    // Try to pick a segment for insert mode
    if (m_EditMode == EditMode::Insert && m_SelectedLineIndex >= 0) {
        int segmentIndex;
        float distance;
        if (PickSegment(ray, m_SelectedLineIndex, segmentIndex, distance)) {
            // Calculate insertion point
            const auto& points = fractureLines[m_SelectedLineIndex].GetPoints();
            Vec3 p0 = points[segmentIndex];
            Vec3 p1 = points[segmentIndex + 1];
            
            // Find closest point on segment
            Vec3 lineDir = p1 - p0;
            float lineLength = lineDir.Length();
            lineDir = lineDir * (1.0f / lineLength);
            
            Vec3 toRayOrigin = ray.origin - p0;
            float projection = toRayOrigin.Dot(lineDir);
            projection = std::max(0.0f, std::min(lineLength, projection));
            
            Vec3 insertPoint = p0 + lineDir * projection;
            InsertPoint(segmentIndex, insertPoint);
            return true;
        }
    }
    
    // Add mode: add point to selected line
    if (m_EditMode == EditMode::Add && m_SelectedLineIndex >= 0) {
        // Intersect ray with a plane at the last point
        const auto& points = fractureLines[m_SelectedLineIndex].GetPoints();
        if (!points.empty()) {
            Vec3 planePoint = points.back();
            Vec3 planeNormal = Vec3(0, 0, 1); // TODO: Use camera forward
            
            float t = RayIntersectPlane(ray, planeNormal, planePoint);
            if (t > 0) {
                Vec3 newPoint = ray.origin + ray.direction * t;
                AddPoint(newPoint);
                return true;
            }
        }
    }
    
    return false;
}

void FractureLineGizmo::OnMouseRelease() {
    m_IsDragging = false;
    m_SelectedPointIndex = -1;
}

void FractureLineGizmo::OnMouseDrag(const Ray& ray, const Camera& camera) {
    if (!m_IsDragging || m_SelectedLineIndex < 0 || m_SelectedPointIndex < 0) return;
    if (!m_SoftBody) return;
    
    // Intersect ray with drag plane
    float t = RayIntersectPlane(ray, m_DragPlaneNormal, m_DragStartPoint);
    if (t > 0) {
        Vec3 newPosition = ray.origin + ray.direction * t;
        MovePoint(m_SelectedPointIndex, newPosition);
    }
}

bool FractureLineGizmo::OnMouseMove(const Ray& ray) {
    if (!m_Enabled || !m_SoftBody) return false;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    
    m_HoveredPointIndex = -1;
    m_HoveredSegmentIndex = -1;
    
    // Check for hovered points
    float closestDistance = std::numeric_limits<float>::max();
    for (int i = 0; i < static_cast<int>(fractureLines.size()); ++i) {
        int pointIndex;
        float distance;
        if (PickPoint(ray, i, pointIndex, distance)) {
            if (distance < closestDistance && i == m_SelectedLineIndex) {
                closestDistance = distance;
                m_HoveredPointIndex = pointIndex;
            }
        }
    }
    
    // Check for hovered segments if no point is hovered
    if (m_HoveredPointIndex < 0 && m_SelectedLineIndex >= 0) {
        int segmentIndex;
        float distance;
        if (PickSegment(ray, m_SelectedLineIndex, segmentIndex, distance)) {
            m_HoveredSegmentIndex = segmentIndex;
        }
    }
    
    return m_HoveredPointIndex >= 0 || m_HoveredSegmentIndex >= 0;
}

void FractureLineGizmo::CreateNewFractureLine() {
    if (!m_SoftBody) return;
    
    FractureLine newLine(0.5f);
    newLine.SetWidth(0.1f);
    m_SoftBody->AddFractureLine(newLine);
    
    m_SelectedLineIndex = static_cast<int>(m_SoftBody->GetFractureLines().size()) - 1;
    m_SelectedPointIndex = -1;
}

void FractureLineGizmo::DeleteSelectedFractureLine() {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return;
    
    m_SoftBody->RemoveFractureLine(m_SelectedLineIndex);
    m_SelectedLineIndex = -1;
    m_SelectedPointIndex = -1;
}

void FractureLineGizmo::SelectFractureLine(int index) {
    if (!m_SoftBody) return;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (index >= 0 && index < static_cast<int>(fractureLines.size())) {
        m_SelectedLineIndex = index;
        m_SelectedPointIndex = -1;
    }
}

void FractureLineGizmo::DeselectAll() {
    m_SelectedLineIndex = -1;
    m_SelectedPointIndex = -1;
}

void FractureLineGizmo::AddPoint(const Vec3& point) {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return;
    
    FractureLine* line = m_SoftBody->GetFractureLine(m_SelectedLineIndex);
    if (line) {
        line->AddPoint(point);
        m_SoftBody->UpdateFractureLine(m_SelectedLineIndex);
    }
}

void FractureLineGizmo::RemovePoint(int pointIndex) {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return;
    
    FractureLine* line = m_SoftBody->GetFractureLine(m_SelectedLineIndex);
    if (line) {
        line->RemovePoint(pointIndex);
        m_SoftBody->UpdateFractureLine(m_SelectedLineIndex);
    }
}

void FractureLineGizmo::InsertPoint(int segmentIndex, const Vec3& point) {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return;
    
    FractureLine* line = m_SoftBody->GetFractureLine(m_SelectedLineIndex);
    if (line) {
        line->InsertPoint(segmentIndex + 1, point);
        m_SoftBody->UpdateFractureLine(m_SelectedLineIndex);
    }
}

void FractureLineGizmo::MovePoint(int pointIndex, const Vec3& newPosition) {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return;
    
    FractureLine* line = m_SoftBody->GetFractureLine(m_SelectedLineIndex);
    if (line) {
        line->SetPoint(pointIndex, newPosition);
        m_SoftBody->UpdateFractureLine(m_SelectedLineIndex);
    }
}

void FractureLineGizmo::SetSelectedLineWeakness(float weakness) {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return;
    
    FractureLine* line = m_SoftBody->GetFractureLine(m_SelectedLineIndex);
    if (line) {
        line->SetWeaknessMultiplier(weakness);
        m_SoftBody->UpdateFractureLine(m_SelectedLineIndex);
    }
}

void FractureLineGizmo::SetSelectedLineWidth(float width) {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return;
    
    FractureLine* line = m_SoftBody->GetFractureLine(m_SelectedLineIndex);
    if (line) {
        line->SetWidth(width);
        m_SoftBody->UpdateFractureLine(m_SelectedLineIndex);
    }
}

float FractureLineGizmo::GetSelectedLineWeakness() const {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return 0.5f;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (m_SelectedLineIndex < static_cast<int>(fractureLines.size())) {
        return fractureLines[m_SelectedLineIndex].GetWeaknessMultiplier();
    }
    return 0.5f;
}

float FractureLineGizmo::GetSelectedLineWidth() const {
    if (!m_SoftBody || m_SelectedLineIndex < 0) return 0.1f;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (m_SelectedLineIndex < static_cast<int>(fractureLines.size())) {
        return fractureLines[m_SelectedLineIndex].GetWidth();
    }
    return 0.1f;
}

bool FractureLineGizmo::PickPoint(const Ray& ray, int lineIndex, int& outPointIndex, float& outDistance) {
    if (!m_SoftBody) return false;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (lineIndex < 0 || lineIndex >= static_cast<int>(fractureLines.size())) return false;
    
    const auto& points = fractureLines[lineIndex].GetPoints();
    float pickRadius = m_PointSize * 2.0f;
    
    float closestDistance = std::numeric_limits<float>::max();
    int closestPoint = -1;
    
    for (size_t i = 0; i < points.size(); ++i) {
        float t = RayClosestPoint(ray, points[i]);
        if (t > 0) {
            Vec3 closestOnRay = ray.origin + ray.direction * t;
            float distance = (closestOnRay - points[i]).Length();
            
            if (distance < pickRadius && t < closestDistance) {
                closestDistance = t;
                closestPoint = static_cast<int>(i);
            }
        }
    }
    
    if (closestPoint >= 0) {
        outPointIndex = closestPoint;
        outDistance = closestDistance;
        return true;
    }
    
    return false;
}

bool FractureLineGizmo::PickSegment(const Ray& ray, int lineIndex, int& outSegmentIndex, float& outDistance) {
    if (!m_SoftBody) return false;
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (lineIndex < 0 || lineIndex >= static_cast<int>(fractureLines.size())) return false;
    
    const auto& points = fractureLines[lineIndex].GetPoints();
    if (points.size() < 2) return false;
    
    float pickRadius = m_LineThickness * 3.0f;
    float closestDistance = std::numeric_limits<float>::max();
    int closestSegment = -1;
    
    for (size_t i = 0; i < points.size() - 1; ++i) {
        Vec3 p0 = points[i];
        Vec3 p1 = points[i + 1];
        Vec3 lineDir = p1 - p0;
        float lineLength = lineDir.Length();
        
        if (lineLength < 0.0001f) continue;
        
        lineDir = lineDir * (1.0f / lineLength);
        
        // Find closest point on segment to ray
        Vec3 toRayOrigin = ray.origin - p0;
        float projection = toRayOrigin.Dot(lineDir);
        projection = std::max(0.0f, std::min(lineLength, projection));
        
        Vec3 closestOnSegment = p0 + lineDir * projection;
        float t = RayClosestPoint(ray, closestOnSegment);
        
        if (t > 0) {
            Vec3 closestOnRay = ray.origin + ray.direction * t;
            float distance = (closestOnRay - closestOnSegment).Length();
            
            if (distance < pickRadius && t < closestDistance) {
                closestDistance = t;
                closestSegment = static_cast<int>(i);
            }
        }
    }
    
    if (closestSegment >= 0) {
        outSegmentIndex = closestSegment;
        outDistance = closestDistance;
        return true;
    }
    
    return false;
}

Vec3 FractureLineGizmo::GetColorForWeakness(float weakness) const {
    // Interpolate between strong (green) and weak (red)
    // weakness = 0.0 means very weak (red)
    // weakness = 1.0 means normal strength (green)
    
    float t = weakness;
    Vec3 color;
    color.x = m_ColorWeak.x * (1.0f - t) + m_ColorStrong.x * t;
    color.y = m_ColorWeak.y * (1.0f - t) + m_ColorStrong.y * t;
    color.z = m_ColorWeak.z * (1.0f - t) + m_ColorStrong.z * t;
    
    return color;
}

// Pattern Library Integration Methods

bool FractureLineGizmo::SaveAsPreset(const std::string& name, const std::string& description) {
    if (!m_PatternLibrary || !m_SoftBody || m_SelectedLineIndex < 0) {
        return false;
    }
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (m_SelectedLineIndex >= static_cast<int>(fractureLines.size())) {
        return false;
    }
    
    const FractureLine& line = fractureLines[m_SelectedLineIndex];
    return m_PatternLibrary->SavePreset(name, line, description);
}

bool FractureLineGizmo::LoadPreset(const std::string& name) {
    if (!m_PatternLibrary || !m_SoftBody) {
        return false;
    }
    
    FractureLine line(0.5f);
    if (!m_PatternLibrary->LoadPreset(name, line)) {
        return false;
    }
    
    // Add loaded fracture line to soft body
    m_SoftBody->AddFractureLine(line);
    
    // Select the newly added line
    m_SelectedLineIndex = static_cast<int>(m_SoftBody->GetFractureLines().size()) - 1;
    m_SelectedPointIndex = -1;
    
    return true;
}

// Tear Pattern Integration Methods

std::unique_ptr<SoftBodyTearPattern> FractureLineGizmo::ConvertToPattern(
    SoftBodyTearPattern::PatternType type)
{
    if (!m_SoftBody || m_SelectedLineIndex < 0) {
        return nullptr;
    }
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (m_SelectedLineIndex >= static_cast<int>(fractureLines.size())) {
        return nullptr;
    }
    
    const FractureLine& line = fractureLines[m_SelectedLineIndex];
    
    // Convert based on requested type
    if (type == SoftBodyTearPattern::PatternType::Straight) {
        return FractureLineToPattern::ToStraightPattern(line);
    } else if (type == SoftBodyTearPattern::PatternType::Curved) {
        float curvature = FractureLineToPattern::CalculateCurvature(line);
        return FractureLineToPattern::ToCurvedPattern(line, curvature);
    }
    
    // Auto-detect pattern type
    type = FractureLineToPattern::EstimatePatternType(line);
    if (type == SoftBodyTearPattern::PatternType::Straight) {
        return FractureLineToPattern::ToStraightPattern(line);
    } else {
        float curvature = FractureLineToPattern::CalculateCurvature(line);
        return FractureLineToPattern::ToCurvedPattern(line, curvature);
    }
}

void FractureLineGizmo::ExecuteTearAlongLine() {
    if (!m_SoftBody || m_SelectedLineIndex < 0) {
        return;
    }
    
    const auto& fractureLines = m_SoftBody->GetFractureLines();
    if (m_SelectedLineIndex >= static_cast<int>(fractureLines.size())) {
        return;
    }
    
    const FractureLine& line = fractureLines[m_SelectedLineIndex];
    const auto& points = line.GetPoints();
    
    if (points.size() < 2) {
        return;
    }
    
    // Convert to pattern
    auto pattern = ConvertToPattern(SoftBodyTearPattern::PatternType::Straight);
    if (!pattern) {
        return;
    }
    
    // Execute tear using first and last points
    Vec3 startPoint = points.front();
    Vec3 endPoint = points.back();
    
    m_SoftBody->TearAlongPattern(*pattern, startPoint, endPoint);
    
    // Clear preview after execution
    m_ShowTearPreview = false;
    m_PreviewAffectedTets.clear();
}

void FractureLineGizmo::ShowTearPreview(bool show) {
    m_ShowTearPreview = show;
    
    if (show && m_SoftBody && m_SelectedLineIndex >= 0) {
        // Update preview - calculate affected tetrahedra
        // This would require access to tetrahedral mesh data
        // For now, just set the flag
        m_PreviewAffectedTets.clear();
    } else {
        m_PreviewAffectedTets.clear();
    }
}

