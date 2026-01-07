-- Rotator Script
-- Access the GameObject transform

local t = 0

function Update(dt)
    t = t + dt
    -- Currently we don't have "this" or "gameObject" exposed to global scope automatically in our simple implementation
    -- But in a real system, we would.
    -- For now, let's just print to verify it runs.
    
    -- In our test implementation, we just run the file.
end

print("Rotator Script Loaded!")
-- Simple test: Create a Vec3
local v = Vec3(1, 2, 3)
print("Vector created: " .. tostring(v))
