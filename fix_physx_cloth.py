#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fix PhysXCloth.cpp file

with open(r'c:\Users\Stefan\Documents\GitHub\game-engine\src\PhysXCloth.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the corrupted end
bad_ending = """            ++it;
        }
    }
}

#endif // USE_PHYSX"""

good_ending = """                outParticles[i].invWeight = 10.0f;
            }
        }
    }
    
    readData->unlock();
}"""

content = content.replace(bad_ending, good_ending)

# Read pattern methods
with open(r'c:\Users\Stefan\Documents\GitHub\game-engine\src\PhysXCloth_pattern_methods.tmp', 'r', encoding='utf-8') as f:
    pattern_code = f.read()

# Write fixed file
with open(r'c:\Users\Stefan\Documents\GitHub\game-engine\src\PhysXCloth.cpp', 'w', encoding='utf-8') as f:
    f.write(content)
    f.write('\n\n')
    f.write(pattern_code)

print("Fixed PhysXCloth.cpp successfully")
