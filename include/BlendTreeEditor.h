#pragma once

#include <memory>
#include <string>
#include <imgui.h>


class Animator;

class BlendTreeEditor {
public:
    BlendTreeEditor();
    
    // Bind to logic
    void SetAnimator(Animator* animator);
    
    // GUI
    void Render();
    
private:
    Animator* m_Animator;
    
    int m_SelectedTreeIndex;
    int m_SelectedTreeType; // 0=None, 1=1D, 2=2D, 3=3D
    
    // Helpers
    void RenderTreeList();
    void RenderTreeEditor1D(int treeIndex);
    void RenderTreeEditor2D(int treeIndex);
    
    // Temporary edit state
    float m_TestParameter1D;
    float m_TestParameter2D[2];
};
