#pragma once

#include <string>
#include <memory>
#include <vector>
#include "GameObject.h"
#include "TextureManager.h"

class GLTFLoader {
public:
    // Load a glTF 2.0 file (text .gltf or binary .glb)
    // Returns the root GameObject of the loaded scene
    static std::shared_ptr<GameObject> Load(const std::string& path, TextureManager* texManager);

private:
    // Helper class is defined in .cpp to avoid exposing tiny_gltf headers
};
