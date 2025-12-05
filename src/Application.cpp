    : m_Running(false)
    , m_LastFrameTime(0.0f)
    , m_FPS(0.0f)
    , m_FrameCount(0.0f)
    , m_FPSTimer(0.0f)
    , m_SelectedObjectIndex(-1)
{
}

Application::~Application() {
}

bool Application::Init() {
    // Create window
    m_Window = std::make_unique<Window>(800, 600, "Game Engine");
    
    if (!m_Window->Init()) {
        std::cerr << "Failed to initialize window" << std::endl;
        return false;
    }

    // Create and initialize renderer
    m_Renderer = std::make_unique<Renderer>();
    if (!m_Renderer->Init()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return false;
    }

    // Create camera
    m_Camera = std::make_unique<Camera>(Vec3(0, 0, 5), 45.0f, 800.0f / 600.0f);
    m_Renderer->SetCamera(m_Camera.get());
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Initialize text rendering
    m_Text = std::make_unique<Text>();
    if (!m_Text->Init("assets/font_atlas.png", 800, 600)) {
        std::cerr << "Failed to initialize text system" << std::endl;
        return false;
    }

    // Initialize ImGui
    m_ImGui = std::make_unique<ImGuiManager>();
    if (!m_ImGui->Init(m_Window->GetGLFWWindow())) {
        std::cerr << "Failed to initialize ImGui" << std::endl;
        return false;
    }

    // Initialize Preview Renderer
    m_PreviewRenderer = std::make_unique<PreviewRenderer>();
    if (!m_PreviewRenderer->Init(512, 512)) {
        std::cerr << "Failed to initialize PreviewRenderer" << std::endl;
        return false;
    }

    m_Running = true;
    return true;
}

void Application::Run() {
    m_LastFrameTime = static_cast<float>(glfwGetTime());
    
    while (m_Running && !m_Window->ShouldClose()) {
        float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - m_LastFrameTime;
        m_LastFrameTime = currentTime;

        Update(deltaTime);
        Render();
        
        m_Window->SwapBuffers();
        m_Window->PollEvents();
    }
}

void Application::Update(float deltaTime) {
    // Calculate FPS
    m_FrameCount++;
    m_FPSTimer += deltaTime;
    if (m_FPSTimer >= 1.0f) {
        m_FPS = m_FrameCount / m_FPSTimer;
        m_FrameCount = 0.0f;
        m_FPSTimer = 0.0f;
    }

    // Update camera with collision detection
    if (m_Camera) {
        Vec3 oldPos = m_Camera->GetPosition();
        m_Camera->ProcessInput(m_Window->GetGLFWWindow(), deltaTime);
        Vec3 newPos = m_Camera->GetPosition();

        // Create player AABB (size 0.5)
        Vec3 playerSize(0.25f, 0.25f, 0.25f); // Half-extents
        AABB playerBounds(newPos - playerSize, newPos + playerSize);

        // Check collision
        if (m_Renderer->CheckCollision(playerBounds)) {
            // Revert position if collision detected
            // Simple response: just revert to old position
            // Ideally we would slide along the wall, but this prevents walking through objects
            m_Camera->SetPosition(oldPos);
        }
    }
    
    // Update scene (sprites, particles, etc.) with deltaTime
    if (m_Renderer) {
        m_Renderer->Update(deltaTime);
    }

    // Scene Management Input
    if (glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_F5) == GLFW_PRESS) {
        m_Renderer->SaveScene("assets/scene.txt");
    }
    if (glfwGetKey(m_Window->GetGLFWWindow(), GLFW_KEY_F9) == GLFW_PRESS) {
        m_Renderer->LoadScene("assets/scene.txt");
    }
    
    // Shader Hot-Reload (Poll every 1.0s)
    static float shaderTimer = 0.0f;
    shaderTimer += deltaTime;
    if (shaderTimer >= 1.0f) {
        m_Renderer->UpdateShaders();
        shaderTimer = 0.0f;
    }
}

void Application::Render() {
    // Clear the screen and depth buffer
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render scene
    m_Renderer->Render();

    // Render UI (HUD)
    if (m_Text && m_Camera) {
        // FPS counter
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(m_FPS));
        m_Text->RenderText(fpsText, 10.0f, 10.0f, 1.0f, Vec3(0.0f, 1.0f, 0.0f));

        // Camera position
        Vec3 camPos = m_Camera->GetPosition();
        std::string posText = "Pos: (" + 
            std::to_string(static_cast<int>(camPos.x)) + ", " +
            std::to_string(static_cast<int>(camPos.y)) + ", " +
            std::to_string(static_cast<int>(camPos.z)) + ")";
        m_Text->RenderText(posText, 10.0f, 30.0f, 1.0f, Vec3(1.0f, 1.0f, 1.0f));
    }

    // Render ImGui Editor UI
    if (m_ImGui) {
        m_ImGui->BeginFrame();
        RenderEditorUI();
        m_ImGui->EndFrame();
    }
}

void Application::RenderEditorUI() {
    // Scene Hierarchy Panel
    ImGui::Begin("Scene Hierarchy");
    
    auto root = m_Renderer->GetRoot();
    if (root) {
        auto& children = root->GetChildren();
        for (size_t i = 0; i < children.size(); ++i) {
            std::string label = children[i]->GetName() + " ##" + std::to_string(i);
            if (ImGui::Selectable(label.c_str(), m_SelectedObjectIndex == static_cast<int>(i))) {
                m_SelectedObjectIndex = static_cast<int>(i);
            }
        }
        
        ImGui::Separator();
        
        if (ImGui::Button("Add Cube")) {
            m_Renderer->AddCube(Transform(Vec3(0, 0, 0)));
        }
        ImGui::SameLine();
        if (ImGui::Button("Add Pyramid")) {
            m_Renderer->AddPyramid(Transform(Vec3(0, 0, 0)));
        }
        
        if (m_SelectedObjectIndex >= 0 && m_SelectedObjectIndex < static_cast<int>(children.size())) {
            if (ImGui::Button("Delete Selected")) {
                m_Renderer->RemoveObject(m_SelectedObjectIndex);
                m_SelectedObjectIndex = -1;
            }
        }
    }
    
    ImGui::Separator();
    
    if (ImGui::Button("Save Scene (F5)")) {
        m_Renderer->SaveScene("assets/scene.txt");
    }
    if (ImGui::Button("Load Scene (F9)")) {
        m_Renderer->LoadScene("assets/scene.txt");
        m_SelectedObjectIndex = -1;
    }
    
    ImGui::End();
    
    // Object Inspector Panel
    if (root && m_SelectedObjectIndex >= 0 && m_SelectedObjectIndex < static_cast<int>(root->GetChildren().size())) {
        ImGui::Begin("Object Inspector");
        
        auto object = root->GetChildren()[m_SelectedObjectIndex];
        Transform& transform = object->GetTransform();
        
        ImGui::Text("Object: %s", object->GetName().c_str());
        ImGui::Separator();
        
        // Position
        ImGui::Text("Position");
        ImGui::DragFloat("X##pos", &transform.position.x, 0.1f);
        ImGui::DragFloat("Y##pos", &transform.position.y, 0.1f);
        ImGui::DragFloat("Z##pos", &transform.position.z, 0.1f);
        
        ImGui::Separator();
        
        // Rotation
        ImGui::Text("Rotation");
        ImGui::DragFloat("X##rot", &transform.rotation.x, 1.0f);
        ImGui::DragFloat("Y##rot", &transform.rotation.y, 1.0f);
        ImGui::DragFloat("Z##rot", &transform.rotation.z, 1.0f);
        
        ImGui::Separator();
        
        // Scale
        ImGui::Text("Scale");
        ImGui::DragFloat("X##scl", &transform.scale.x, 0.1f, 0.1f, 10.0f);
        ImGui::DragFloat("Y##scl", &transform.scale.y, 0.1f, 0.1f, 10.0f);
        ImGui::DragFloat("Z##scl", &transform.scale.z, 0.1f, 0.1f, 10.0f);
        
        ImGui::Separator();

        // Material Inspector
        auto mat = object->GetMaterial();
        if (mat) {
            // Render and display preview
            if (m_PreviewRenderer) {
                // Create a simple shader for preview (we'll use a basic approach)
                // For now, render the preview - the PreviewRenderer will handle shader internally
                m_PreviewRenderer->RenderPreview(object.get(), nullptr);
                
                // Display preview image
                ImGui::Text("Material Preview");
                ImVec2 previewSize(256, 256);
                ImGui::Image(
                    (void*)(intptr_t)m_PreviewRenderer->GetTextureID(),
                    previewSize,
                    ImVec2(0, 1), // UV coordinates flipped for OpenGL
                    ImVec2(1, 0)
                );
                ImGui::Separator();
            }
            
            ImGui::Text("Material Properties");
            ImGui::Separator();
            
            // Basic Colors
            if (ImGui::CollapsingHeader("Colors", ImGuiTreeNodeFlags_DefaultOpen)) {
                Vec3 ambient = mat->GetAmbient();
                if (ImGui::ColorEdit3("Ambient", &ambient.x)) {
                    mat->SetAmbient(ambient);
                }
                
                Vec3 diffuse = mat->GetDiffuse();
                if (ImGui::ColorEdit3("Diffuse", &diffuse.x)) {
                    mat->SetDiffuse(diffuse);
                }
                
                Vec3 specular = mat->GetSpecular();
                if (ImGui::ColorEdit3("Specular", &specular.x)) {
                    mat->SetSpecular(specular);
                }
                
                Vec3 emissive = mat->GetEmissiveColor();
                if (ImGui::ColorEdit3("Emissive", &emissive.x)) {
                    mat->SetEmissiveColor(emissive);
                }
            }
            
            // PBR Properties
            if (ImGui::CollapsingHeader("PBR Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
                float shininess = mat->GetShininess();
                if (ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f)) {
                    mat->SetShininess(shininess);
                }
                
                float roughness = mat->GetRoughness();
                if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f)) {
                    mat->SetRoughness(roughness);
                }
                
                float metallic = mat->GetMetallic();
                if (ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f)) {
                    mat->SetMetallic(metallic);
                }
                
                float heightScale = mat->GetHeightScale();
                if (ImGui::SliderFloat("Height Scale", &heightScale, 0.0f, 1.0f)) {
                    mat->SetHeightScale(heightScale);
                }
                
                float opacity = mat->GetOpacity();
                if (ImGui::SliderFloat("Opacity", &opacity, 0.0f, 1.0f)) {
                    mat->SetOpacity(opacity);
                }
                
                bool isTransparent = mat->IsTransparent();
                if (ImGui::Checkbox("Transparent", &isTransparent)) {
                    mat->SetIsTransparent(isTransparent);
                }
            }
            
            // Texture Maps
            if (ImGui::CollapsingHeader("Texture Maps")) {
                auto texManager = m_Renderer->GetTextureManager();
                if (texManager) {
                    auto textureNames = texManager->GetTextureNames();
                    
                    // Helper lambda for texture selection
                    auto renderTextureCombo = [&](const char* label, std::shared_ptr<Texture> currentTex, auto setter) {
                        std::string currentName = currentTex ? "Loaded Texture" : "None";
                        if (ImGui::BeginCombo(label, currentName.c_str())) {
                            if (ImGui::Selectable("None", !currentTex)) {
                                (mat.get()->*setter)(nullptr);
                            }
                            for (const auto& name : textureNames) {
                                bool isSelected = (currentTex && currentTex == texManager->GetTexture(name));
                                if (ImGui::Selectable(name.c_str(), isSelected)) {
                                    (mat.get()->*setter)(texManager->GetTexture(name));
                                }
                            }
                            ImGui::EndCombo();
                        }
                    };
                    
                    renderTextureCombo("Albedo/Diffuse", mat->GetTexture(), &Material::SetTexture);
                    renderTextureCombo("Normal Map", mat->GetNormalMap(), &Material::SetNormalMap);
                    renderTextureCombo("Specular Map", mat->GetSpecularMap(), &Material::SetSpecularMap);
                    renderTextureCombo("Roughness Map", mat->GetRoughnessMap(), &Material::SetRoughnessMap);
                    renderTextureCombo("Metallic Map", mat->GetMetallicMap(), &Material::SetMetallicMap);
                    renderTextureCombo("AO Map", mat->GetAOMap(), &Material::SetAOMap);
                    renderTextureCombo("ORM Map", mat->GetORMMap(), &Material::SetORMMap);
                    renderTextureCombo("Height Map", mat->GetHeightMap(), &Material::SetHeightMap);
                    renderTextureCombo("Emissive Map", mat->GetEmissiveMap(), &Material::SetEmissiveMap);
                }
            }
            
            // Material Presets
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Material Presets")) {
                static char presetName[128] = "my_material";
                
                // Save Preset
                ImGui::InputText("Preset Name", presetName, sizeof(presetName));
                if (ImGui::Button("Save Preset")) {
                    std::string filepath = "assets/materials/" + std::string(presetName) + ".mat";
                    if (mat->SaveToFile(filepath)) {
                        std::cout << "Preset saved successfully!" << std::endl;
                    }
                }
                
                ImGui::Separator();
                
                // Load Preset - scan for .mat files
                static std::vector<std::string> presetFiles;
                static int selectedPreset = -1;
                
                if (ImGui::Button("Refresh Presets")) {
                    presetFiles.clear();
                    selectedPreset = -1;
                    
                    // Simple directory scan (Windows-specific for now)
                    #ifdef _WIN32
                    WIN32_FIND_DATAA findData;
                    HANDLE hFind = FindFirstFileA("assets/materials/*.mat", &findData);
                    if (hFind != INVALID_HANDLE_VALUE) {
                        do {
                            if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                                presetFiles.push_back(findData.cFileName);
                            }
                        } while (FindNextFileA(hFind, &findData));
                        FindClose(hFind);
                    }
                    #endif
                }
                
                if (!presetFiles.empty()) {
                    ImGui::Text("Available Presets:");
                    for (size_t i = 0; i < presetFiles.size(); ++i) {
                        if (ImGui::Selectable(presetFiles[i].c_str(), selectedPreset == (int)i)) {
                            selectedPreset = (int)i;
                        }
                    }
                    
                    if (selectedPreset >= 0 && ImGui::Button("Load Selected Preset")) {
                        std::string filepath = "assets/materials/" + presetFiles[selectedPreset];
                        if (mat->LoadFromFile(filepath, m_Renderer->GetTextureManager())) {
                            std::cout << "Preset loaded successfully!" << std::endl;
                        }
                    }
                } else {
                    ImGui::TextDisabled("No presets found. Click 'Refresh Presets'.");
                }
            }
        }

        ImGui::End();
    }

    // Light Inspector Panel
    ImGui::Begin("Light Inspector");
    
    auto& lights = m_Renderer->GetLights();
    static int selectedLightIndex = -1;

    if (ImGui::Button("Add Light")) {
        m_Renderer->AddLight(Light(Vec3(0, 5, 0)));
    }
    
    bool showCascades = m_Renderer->GetShowCascades();
    if (ImGui::Checkbox("Show CSM Cascades", &showCascades)) {
        m_Renderer->SetShowCascades(showCascades);
    }
    
    float fadeStart = m_Renderer->GetShadowFadeStart();
    if (ImGui::SliderFloat("Shadow Fade Start", &fadeStart, 10.0f, 100.0f)) {
        m_Renderer->SetShadowFadeStart(fadeStart);
    }
    
    float fadeEnd = m_Renderer->GetShadowFadeEnd();
    if (ImGui::SliderFloat("Shadow Fade End", &fadeEnd, 10.0f, 100.0f)) {
        m_Renderer->SetShadowFadeEnd(fadeEnd);
    }
    
    ImGui::Separator();

    for (size_t i = 0; i < lights.size(); ++i) {
        std::string label = "Light " + std::to_string(i);
        if (ImGui::Selectable(label.c_str(), selectedLightIndex == static_cast<int>(i))) {
            selectedLightIndex = static_cast<int>(i);
        }
    }

    if (selectedLightIndex >= 0 && selectedLightIndex < static_cast<int>(lights.size())) {
        ImGui::Separator();
        Light& light = lights[selectedLightIndex];
        
        ImGui::Text("Light Properties");
        
        // Light Type
        const char* lightTypes[] = { "Directional", "Point", "Spot" };
        int currentType = static_cast<int>(light.type);
        if (ImGui::Combo("Type", &currentType, lightTypes, IM_ARRAYSIZE(lightTypes))) {
            light.type = static_cast<LightType>(currentType);
        }

        bool castsShadows = light.castsShadows;
        if (ImGui::Checkbox("Cast Shadows", &castsShadows)) {
            light.castsShadows = castsShadows;
        }
        if (light.castsShadows) {
            if (light.type == LightType::Point) {
                ImGui::SliderFloat("Shadow Softness", &light.shadowSoftness, 1.0f, 5.0f);
            } else {
                ImGui::SliderFloat("Light Size (PCSS)", &light.lightSize, 0.0f, 5.0f);
            }
        }

        if (light.type != LightType::Directional) {
            ImGui::DragFloat("Range", &light.range, 0.5f, 1.0f, 100.0f);
        }
        
        ImGui::DragFloat3("Position", &light.position.x, 0.1f);
        if (light.type != LightType::Point) {
            ImGui::DragFloat3("Direction", &light.direction.x, 0.1f);
        }
        
        ImGui::ColorEdit3("Color", &light.color.x);
        ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 10.0f);
        
        if (light.type != LightType::Directional) {
            ImGui::Text("Attenuation (Advanced)");
            ImGui::DragFloat("Constant", &light.constant, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Linear", &light.linear, 0.001f, 0.0f, 1.0f);
            ImGui::DragFloat("Quadratic", &light.quadratic, 0.001f, 0.0f, 1.0f);
        }
        
        if (light.type == LightType::Spot) {
            ImGui::Text("Spotlight");
            ImGui::DragFloat("Cutoff", &light.cutOff, 0.1f, 0.0f, 90.0f);
            ImGui::DragFloat("Outer Cutoff", &light.outerCutOff, 0.1f, 0.0f, 90.0f);
        }
        
        if (ImGui::Button("Delete Light")) {
            m_Renderer->RemoveLight(selectedLightIndex);
            selectedLightIndex = -1;
        }
    }

    ImGui::End();
    
    // Post-Processing Panel
    ImGui::Begin("Post-Processing");
    
    auto postProcessing = m_Renderer->GetPostProcessing();
    if (postProcessing) {
        // Bloom settings
        ImGui::Text("Bloom");
        bool bloomEnabled = postProcessing->IsBloomEnabled();
        if (ImGui::Checkbox("Enable Bloom", &bloomEnabled)) {
            postProcessing->SetBloomEnabled(bloomEnabled);
        }
        
        if (bloomEnabled) {
            float bloomIntensity = postProcessing->GetBloomIntensity();
            if (ImGui::SliderFloat("Bloom Intensity", &bloomIntensity, 0.0f, 2.0f)) {
                postProcessing->SetBloomIntensity(bloomIntensity);
            }
            
            float bloomThreshold = postProcessing->GetBloomThreshold();
            if (ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.0f, 5.0f)) {
                postProcessing->SetBloomThreshold(bloomThreshold);
            }
        }
        
        ImGui::Separator();
        
        // Tone mapping settings
        ImGui::Text("Tone Mapping");
        const char* toneMappingModes[] = { "Reinhard", "ACES Filmic" };
        int toneMappingMode = postProcessing->GetToneMappingMode();
        if (ImGui::Combo("Mode", &toneMappingMode, toneMappingModes, IM_ARRAYSIZE(toneMappingModes))) {
            postProcessing->SetToneMappingMode(toneMappingMode);
        }
        
        float exposure = postProcessing->GetExposure();
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f)) {
            postProcessing->SetExposure(exposure);
        }
        
        float gamma = postProcessing->GetGamma();
        if (ImGui::SliderFloat("Gamma", &gamma, 1.8f, 2.4f)) {
            postProcessing->SetGamma(gamma);
        }
        
        ImGui::Separator();
        
        // SSAO settings
        ImGui::Text("SSAO (Ambient Occlusion)");
        bool ssaoEnabled = m_Renderer->GetSSAOEnabled();
        if (ImGui::Checkbox("Enable SSAO", &ssaoEnabled)) {
            m_Renderer->SetSSAOEnabled(ssaoEnabled);
        }
        
        if (ssaoEnabled) {
            auto ssao = m_Renderer->GetSSAO();
            if (ssao) {
                float radius = ssao->GetRadius();
                if (ImGui::SliderFloat("SSAO Radius", &radius, 0.1f, 2.0f)) {
                    ssao->SetRadius(radius);
                }
                
                float bias = ssao->GetBias();
                if (ImGui::SliderFloat("SSAO Bias", &bias, 0.001f, 0.1f)) {
                    ssao->SetBias(bias);
                }
            }
        }
    }
    
    ImGui::End();
}
