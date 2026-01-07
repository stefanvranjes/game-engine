# Rotator Script (Python)
import time

print("Rotator Script Loaded via Python!")

# Example usage
v = Vec3(1.0, 2.0, 3.0)
print(f"Created Vector: {v}")

# Create a GameObject
go = GameObject("PythonObject")
t = go.get_transform()
t.position = Vec3(10, 0, 0)
print(f"GameObject Position: {t.position}")
