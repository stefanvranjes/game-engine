#include "AudioListener.h"
#include "AudioSystem.h"

AudioListener::AudioListener() 
    : m_Enabled(true)
{
}

AudioListener::~AudioListener() {
    // If this was the active listener, maybe clear it?
    // Use GetActiveListener to check, but AudioSystem handles raw pointers.
    // For safety, let's just leave it. The system updates every frame anyway.
}

void AudioListener::MakeActive() {
    AudioSystem::Get().SetActiveListener(this);
}
