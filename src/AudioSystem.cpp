#include "AudioSystem.h"
#include <iostream>

// Define implementation only in this .cpp file
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

AudioSystem& AudioSystem::Get() {
    static AudioSystem instance;
    return instance;
}

void AudioSystem::SetActiveListener(AudioListener* listener) {
    m_ActiveListener = listener;
}

AudioSystem::AudioSystem() {
}

AudioSystem::~AudioSystem() {
    Shutdown();
}

bool AudioSystem::Initialize() {
    if (m_initialized) return true;

    ma_result result;
    ma_engine_config engineConfig = ma_engine_config_init();

    result = ma_engine_init(&engineConfig, &m_engine);
    if (result != MA_SUCCESS) {
        std::cerr << "Failed to initialize audio engine." << std::endl;
        return false;
    }

    // Initialize Reverb Node
    // We get the node graph from the engine
    ma_node_graph* nodeGraph = ma_engine_get_node_graph(&m_engine);
    
    ma_reverb_node_config reverbConfig = ma_reverb_node_config_init(nodeGraph->pResourceManager->config.channels, nodeGraph->pResourceManager->config.sampleRate);
    reverbConfig.roomSize = 0.5f; // Default
    reverbConfig.damping = 0.5f;
    reverbConfig.wetVolume = 0.3f; // Slight reverb by default
    reverbConfig.dryVolume = 1.0f;
    
    result = ma_reverb_node_init(nodeGraph, &reverbConfig, NULL, &m_reverbNode);
    if (result != MA_SUCCESS) {
        // Fallback?
    }

    // Connect Reverb to Engine Endpoint (Master Out)
    // ma_node_attach_output_bus(&m_reverbNode, 0, ma_engine_get_endpoint(&m_engine), 0);
    // Actually, usually we route reverb output to master, but input comes from group.
    
    // Initialize World Sound Group
    // We want this group to output to the Reverb Node INSTEAD of the Endpoint.
    // However, ma_sound_group_init defaults to endpoint.
    // Use flags or re-attach later?
    
    result = ma_sound_group_init(&m_engine, 0, NULL, &m_worldGroup);
    if (result != MA_SUCCESS) {
        return false;
    }

    // Initialize Music Group (routes to Endpoint by default, which is what we want)
    result = ma_sound_group_init(&m_engine, 0, NULL, &m_musicGroup);
    if (result != MA_SUCCESS) {
        return false;
    }
    
    // Re-route World Group Output -> Reverb Node Input
    // Note: miniaudio nodes are single-input single-output (bus-wise) usually unless configured explicitly.
    // We connect Group's internal node to Reverb.
    ma_node* groupNode = ma_sound_group_get_node(&m_worldGroup);
    ma_node* endpoint = ma_engine_get_endpoint(&m_engine);
    
    // 1. Detach Group from Endpoint (it was auto-attached)
    ma_node_detach_output_bus(groupNode, 0);
    
    // 2. Attach Group to Reverb
    ma_node_attach_output_bus(groupNode, 0, &m_reverbNode, 0);
    
    // 3. Attach Reverb to Endpoint
    ma_node_attach_output_bus(&m_reverbNode, 0, endpoint, 0);

    m_initialized = true;
    std::cout << "Audio System Initialized." << std::endl;
    return true;
}

void AudioSystem::Shutdown() {
    if (!m_initialized) return;

    ma_reverb_node_uninit(&m_reverbNode, NULL); // Clean up node
    ma_sound_group_uninit(&m_worldGroup);       // Clean up group
    ma_sound_group_uninit(&m_musicGroup);       // Clean up music group
    ma_engine_uninit(&m_engine);
    m_initialized = false;
}

ma_engine* AudioSystem::GetEngine() {
    return &m_engine;
}

void AudioSystem::SetMasterVolume(float volume) {
    if (!m_initialized) return;
    ma_engine_set_volume(&m_engine, volume);
}

void AudioSystem::SetListenerPosition(const Vec3& position) {
    if (!m_initialized) return;
    ma_engine_listener_set_position(&m_engine, 0, position.x, position.y, position.z);
}

void AudioSystem::SetListenerDirection(const Vec3& forward) {
    if (!m_initialized) return;
    ma_engine_listener_set_direction(&m_engine, 0, forward.x, forward.y, forward.z);
}

void AudioSystem::SetListenerVelocity(const Vec3& velocity) {
    if (!m_initialized) return;
    ma_engine_listener_set_velocity(&m_engine, 0, velocity.x, velocity.y, velocity.z);
}

void AudioSystem::SetSFXVolume(float volume) {
    if (!m_initialized) return;
    ma_sound_group_set_volume(&m_worldGroup, volume);
}

void AudioSystem::SetMusicVolume(float volume) {
    if (!m_initialized) return;
    ma_sound_group_set_volume(&m_musicGroup, volume);
}
