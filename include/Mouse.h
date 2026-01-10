#pragma once

/**
 * @brief Simple mouse input structure
 */
struct Mouse {
    float x;              // Mouse X position
    float y;              // Mouse Y position
    bool leftButtonPressed;
    bool rightButtonPressed;
    bool middleButtonPressed;
    float scrollDelta;
    
    Mouse() 
        : x(0), y(0)
        , leftButtonPressed(false)
        , rightButtonPressed(false)
        , middleButtonPressed(false)
        , scrollDelta(0)
    {}
};
