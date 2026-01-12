void Application::LoadGpuTestScene() {
    // Clear existing scene
    m_Renderer->GetRoot()->GetChildren().clear();
    m_SelectedObjectIndex = -1;
    
    // Create Floor
    {
        auto floor = std::make_shared<GameObject>("Floor");
        floor->GetTransform().position = Vec3(0, -1, 0);
        floor->GetTransform().scale = Vec3(50, 1, 50);
        floor->SetMesh(Mesh::CreateCube());
        
        auto mat = std::make_shared<Material>();
        mat->SetDiffuse(Vec3(0.5f, 0.5f, 0.5f));
        floor->SetMaterial(mat);
        
        // PhysX Static Body
        auto body = std::make_shared<PhysXRigidBody>(m_PhysXBackend.get());
        PhysicsCollisionShape shape;
        shape.SetBox(Vec3(25, 0.5f, 25)); // Half-extents
        body->Initialize(BodyType::Static, 0.0f, shape.GetShapePointer()); 
        // Need to ensure shape pointer is compatible or refactor helper.
        // Actually PhysXRigidBody::Initialize takes shared_ptr<IPhysicsShape>.
        // PhysicsCollisionShape wraps it but might expose a raw pointer or similar.
        // Let's create IPhysicsShape directly for cleaner code or use existing wrappers.
        
        // Simpler: Use existing pattern if available. 
        // Assuming PhysicsCollisionShape::GetShapePointer() returns shared_ptr<IPhysicsShape> compatibility?
        // Wait, PhysicsCollisionShape is usually for Bullet. 
        // I need to use PhysXShape if I am bypassing the generic PhysicsSystem wrapper?
        // Or better: Use PhysicsSystem adapter if I created one?
        // For now, let's just make a PhysXShape directly if we can't easy-wrap.
        
        // Create PhysX Box Shape
        // Since we don't have a factory easily available here, let's skip shape details and assume we can make one?
        // Actually, let's use the generic IPhysicsShape
        // Oops, I can't easily create a PhysXShape without the backend methods.
        // Let's assume for this test we can use what we have or just spawn a lot of bodies without shapes? No.
        
        // Correction: PhysicsCollisionShape might internally hold a Bullet shape, not PhysX.
        // Using `PhysXShape` class.
    }
    
    // Actually, to make this robust, I'll rely on PhysXShape.
    // I need to include "PhysXShape.h"
}
