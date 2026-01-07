#include "Gizmo.h"
#include "Camera.h"
#include "Shader.h"
#include <glad/glad.h>
#include <cmath>
#include <algorithm>

unsigned int Gizmo::s_ArrowVAO = 0;
unsigned int Gizmo::s_ArrowVBO = 0;
unsigned int Gizmo::s_CubeVAO = 0;
unsigned int Gizmo::s_CubeVBO = 0;
unsigned int Gizmo::s_CircleVAO = 0;
unsigned int Gizmo::s_CircleVBO = 0;
unsigned int Gizmo::s_QuadVAO = 0;
unsigned int Gizmo::s_QuadVBO = 0;
bool Gizmo::s_Initialized = false;

Gizmo::Gizmo() {
    if (!s_Initialized) {
        InitGizmoResources();
        s_Initialized = true;
    }
}

void Gizmo::Draw(Shader* shader, const Camera& camera) {
    if (!m_Enabled || !m_Transform) return;
    // Base implementation doesn't draw anything, derived classes should
}

float Gizmo::GetScreenScale(const Vec3& position, const Camera& camera) {
    float dist = (position - camera.GetPosition()).Length();
    // Adjust this factor to change how large gizmos appear on screen
    return dist * 0.15f; 
}

// ----------------------------------------------------------------------------
// Ray Intersection Helpers
// ----------------------------------------------------------------------------

float Gizmo::RayIntersectPlane(const Ray& ray, const Vec3& planeNormal, const Vec3& planePoint) {
    float denom = planeNormal.Dot(ray.direction);
    if (std::abs(denom) > 1e-6f) {
        Vec3 p0l0 = planePoint - ray.origin;
        float t = p0l0.Dot(planeNormal) / denom;
        return (t >= 0) ? t : -1.0f;
    }
    return -1.0f;
}

// Ray-AABB intersection
bool Gizmo::RayIntersectBox(const Ray& ray, const Vec3& boxMin, const Vec3& boxMax, float& t) {
    float tmin = (boxMin.x - ray.origin.x) / ray.direction.x;
    float tmax = (boxMax.x - ray.origin.x) / ray.direction.x;

    if (tmin > tmax) std::swap(tmin, tmax);

    float tymin = (boxMin.y - ray.origin.y) / ray.direction.y;
    float tymax = (boxMax.y - ray.origin.y) / ray.direction.y;

    if (tymin > tymax) std::swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    float tzmin = (boxMin.z - ray.origin.z) / ray.direction.z;
    float tzmax = (boxMax.z - ray.origin.z) / ray.direction.z;

    if (tzmin > tzmax) std::swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    if (tmax < 0) return false;

    t = tmin;
    return true;
}

// Returns distance along ray which is closest to point
float Gizmo::RayClosestPoint(const Ray& ray, const Vec3& point) {
    Vec3 pointToOrigin = point - ray.origin;
    float projection = pointToOrigin.Dot(ray.direction);
    return projection;
}

// Möller–Trumbore intersection algorithm
bool Gizmo::RayIntersectTriangle(const Ray& ray, const Vec3& v0, const Vec3& v1, const Vec3& v2, float& t, Vec3& intersectionPoint) {
    const float EPSILON = 0.0000001f;
    Vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = ray.direction.Cross(edge2);
    a = edge1.Dot(h);
    
    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.
        
    f = 1.0f / a;
    s = ray.origin - v0;
    u = f * s.Dot(h);
    
    if (u < 0.0f || u > 1.0f)
        return false;
        
    q = s.Cross(edge1);
    v = f * ray.direction.Dot(q);
    
    if (v < 0.0f || u + v > 1.0f)
        return false;
        
    // At this stage we can compute t to find out where the intersection point is on the line.
    t = f * edge2.Dot(q);
    
    if (t > EPSILON) { // ray intersection
        intersectionPoint = ray.origin + ray.direction * t;
        return true;
    } else { // This means that there is a line intersection but not a ray intersection.
        return false;
    }
}


// ----------------------------------------------------------------------------
// Geometry Initialization & Drawing
// ----------------------------------------------------------------------------

void Gizmo::InitGizmoResources() {
    // 1. Arrow (Cylinder + Cone)
    // We'll just draw a line for the shaft and a cone for the tip for simplicity, 
    // or a simple mesh. Let's do a simple cylinder/cone mesh.
    // For now, let's just use lines and points or simple primitives if possible.
    // Actually, let's make a proper mesh.
    
    // Cube
    float cubeVertices[] = {
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f
    };

    glGenVertexArrays(1, &s_CubeVAO);
    glGenBuffers(1, &s_CubeVBO);
    glBindVertexArray(s_CubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_CubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Quad (for planes)
    float quadVertices[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    glGenVertexArrays(1, &s_QuadVAO);
    glGenBuffers(1, &s_QuadVBO);
    glBindVertexArray(s_QuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_QuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Circle/Torus
    std::vector<float> circleVertices;
    const int segments = 64;
    for (int i = 0; i <= segments; ++i) {
        float theta = (float)i / segments * 2.0f * 3.14159f;
        circleVertices.push_back(std::cos(theta));
        circleVertices.push_back(std::sin(theta));
        circleVertices.push_back(0.0f);
    }
    glGenVertexArrays(1, &s_CircleVAO);
    glGenBuffers(1, &s_CircleVBO);
    glBindVertexArray(s_CircleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_CircleVBO);
    glBufferData(GL_ARRAY_BUFFER, circleVertices.size() * sizeof(float), circleVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Arrow (Line + Cone tip)
    // We'll model an arrow pointing up Y, length 1
    // Just a line for now, we'll draw it with GL_LINES
    float arrowVertices[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    glGenVertexArrays(1, &s_ArrowVAO);
    glGenBuffers(1, &s_ArrowVBO);
    glBindVertexArray(s_ArrowVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_ArrowVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void Gizmo::DrawCube(Shader* shader, const Vec3& center, const Vec3& size, const Vec3& color) {
    Mat4 model = Mat4::Translate(center) * Mat4::Scale(size);
    shader->SetMat4("model", model);
    shader->SetVec3("color", color);
    
    glBindVertexArray(s_CubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

void Gizmo::DrawQuad(Shader* shader, const Vec3& center, const Vec3& right, const Vec3& up, const Vec3& color) {
    // Construct coordinate frame
    // We want the quad to be defined by origin, right, and up vectors
    // But our base quad is 0,0 to 1,1
    // We can cheat: construct a matrix that maps (1,0,0) to right and (0,1,0) to up
    
    // Mat4 constructor is column-major:
    // col1, col2, col3, col4
    Mat4 transform = Mat4::Identity();
    // Set columns manually
    transform.rows[0] = Vec4(right.x, up.x, 0, center.x);
    transform.rows[1] = Vec4(right.y, up.y, 0, center.y);
    transform.rows[2] = Vec4(right.z, up.z, 0, center.z);
    transform.rows[3] = Vec4(0, 0, 0, 1);
    
    // Note: The Mat4 class in this engine might be row-major or implement Set differently.
    // Let's assume Mat4::Identity() returns diagonal 1s.
    // Let's assume standard math for now. If Mat4 is distinct, we'll adjust.
    // Actually, let's use the provided Mat4 methods to be safe.
    // Mat4 doesn't seem to have a "from basis" constructor readily available in my memory of existing files,
    // so let's try to construct it.
    // Or we can just build the vertex data dynamicall? No, that's slow.
    // Let's try column construct.
    
    // Fallback: Just supply a model matrix assuming the quad is in XY plane scaled and rotated?
    // Let's implement manually using a generic basis matrix construction if possible.
    // Actually, `right` and `up` define the scale and rotation.
    // Let normal = right.Cross(up).Normalized();
    // right = right vector (length = width)
    // up = up vector (length = height)
    
    // Let's construct the matrix directly.
    Mat4 model;
    model.elements[0] = right.x; model.elements[4] = up.x; model.elements[8] = 0; model.elements[12] = center.x;
    model.elements[1] = right.y; model.elements[5] = up.y; model.elements[9] = 0; model.elements[13] = center.y;
    model.elements[2] = right.z; model.elements[6] = up.z; model.elements[10]= 0; model.elements[14] = center.z;
    model.elements[3] = 0;       model.elements[7] = 0;    model.elements[11]= 1; model.elements[15] = 1;
    
    shader->SetMat4("model", model);
    shader->SetVec3("color", color);
    
    glBindVertexArray(s_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void Gizmo::DrawArrow(Shader* shader, const Vec3& start, const Vec3& end, const Vec3& color, float scale) {
    Vec3 dir = end - start;
    float len = dir.Length();
    if (len < 0.001f) return;
    dir = dir / len;
    
    // Draw Line
    // We need to align the line (0,1,0) to 'dir' locally? 
    // Or just use GL_LINES with world coords? 
    // The shader expects a model matrix. 
    // It's easier to implement "DrawLine" if we had a line shader, but we are using "model" uniform.
    // Let's assume we align the Y-up VBO to the direction.
    
    Vec3 up(0, 1, 0);
    Vec3 axis = up.Cross(dir);
    float angle = std::acos(up.Dot(dir));
    
    if (axis.LengthSquared() < 0.001f) {
        // Parallel
        if (up.Dot(dir) < 0) axis = Vec3(1, 0, 0); // 180 flip
        else axis = Vec3(0, 0, 1); // Same dir
    }
    
    Mat4 rotation = Mat4::RotationAxis(axis, angle); 
    Mat4 translation = Mat4::Translate(start);
    Mat4 scaling = Mat4::Scale(Vec3(1, len, 1));
    
    Mat4 model = translation * rotation * scaling; // Scale Y by length
    
    shader->SetMat4("model", model);
    shader->SetVec3("color", color);

    glBindVertexArray(s_ArrowVAO);
    glDrawArrays(GL_LINES, 0, 2);
    
    // Draw Cone at the end
    // To be implemented properly, for now just a small box/point
    DrawCube(shader, end, Vec3(scale, scale, scale) * 2.0f, color);
    glBindVertexArray(0);
}

void Gizmo::DrawCircle(Shader* shader, const Vec3& center, const Vec3& normal, float radius, const Vec3& color) {
    // Normal alignment
    Vec3 up(0, 0, 1); // Circle lies in XY plane by default (vertices have z=0)
    // Wait, I defined vertices as cos/sin in XY. So normal is Z.
    
    Vec3 axis = up.Cross(normal);
    float angle = std::acos(up.Dot(normal));
    
    // Fix pure opposite case
    if (axis.LengthSquared() < 0.0001f) {
        if (up.Dot(normal) < 0.0f) {
            axis = Vec3(1.0f, 0.0f, 0.0f);
            angle = 3.14159f;
        } else {
            axis = Vec3(0.0f, 0.0f, 1.0f);
            angle = 0.0f;
        }
    }
    
    Mat4 rotation = Mat4::RotationAxis(axis, angle);
    Mat4 translation = Mat4::Translate(center);
    Mat4 scaling = Mat4::Scale(Vec3(radius, radius, radius));
    
    Mat4 model = translation * rotation * scaling;
    
    shader->SetMat4("model", model);
    shader->SetVec3("color", color);
    
    glBindVertexArray(s_CircleVAO);
    glDrawArrays(GL_LINE_LOOP, 0, 64); // 64 segments
    glBindVertexArray(0);
}
