import gym
from gym import spaces
import numpy as np
import game_engine as ge # The C++ module we bound

class PhysXRagdollEnv(gym.Env):
    """
    OpenAI Gym Environment for a PhysX Ragdoll.
    
    This environment allows an RL agent to control a ragdoll using joint torques.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, ragdoll, suspension_time=0.016):
        super(PhysXRagdollEnv, self).__init__()
        
        self.ragdoll = ragdoll
        self.dt = suspension_time
        
        # Define Observation Space
        # Rigid Body State: Position (3), Rotation (4), LinVel (3), AngVel (3) = 13 per link
        # For a simple ragdoll (e.g. 10 bones), that's ~130 dimensions.
        # We'll assume a fixed number of links for simplicity or dynamic.
        # Let's try to discover links.
        self.link_names = ["Hips", "Spine", "Head", "LeftArm", "RightArm", "LeftLeg", "RightLeg"] # Example
        self.num_links = len(self.link_names)
        self.obs_dim = self.num_links * 13 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Define Action Space
        # Torque for each link (3 DOF)
        self.action_space = spaces.Box(
            low=-1000.0, high=1000.0, shape=(self.num_links * 3,), dtype=np.float32
        )
        
    def reset(self):
        # Reset simulation state
        # In a real engine, we might reload the scene or teleport the ragdoll
        self.ragdoll.set_state(ge.RagdollState.Kinematic) # Snap to start
        # self.ragdoll.match_animation(...) # If we bound this
        self.ragdoll.set_state(ge.RagdollState.Active) # Enable control
        
        return self._get_obs()
        
    def step(self, action):
        # 1. Apply Actions (Torques)
        for i, name in enumerate(self.link_names):
            link = self.ragdoll.get_link(name)
            if link:
                # Extract torque for this link
                tx = action[i*3 + 0]
                ty = action[i*3 + 1]
                tz = action[i*3 + 2]
                link.add_torque(ge.Vec3(tx, ty, tz))
        
        # 2. Step Simulation
        # Requires engine hook. Assuming ge.step_physics() exists or we rely on main loop.
        # For this wrapper, we assume the engine drives the loop if embedded, 
        # or we call a bound specific step function.
        if hasattr(ge, "step_physics"):
            ge.step_physics(self.dt)
        
        # 3. Get Observation
        obs = self._get_obs()
        
        # 4. Calculate Reward
        # Example: Keep Head up
        reward = 0.0
        head_link = self.ragdoll.get_link("Head")
        if head_link:
            pos = head_link.get_position()
            reward = pos.y # Simple reward: higher is better
            
        # 5. Check Done
        done = False
        if reward < 0.5: # Fell down
            done = True
            
        return obs, reward, done, {}
        
    def _get_obs(self):
        obs = []
        for name in self.link_names:
            link = self.ragdoll.get_link(name)
            if link:
                pos = link.get_position()
                rot = link.get_rotation() # Quat
                lin_vel = link.get_linear_velocity()
                ang_vel = link.get_angular_velocity()
                
                obs.extend([pos.x, pos.y, pos.z])
                obs.extend([rot.x, rot.y, rot.z, rot.w]) # Check Quat/Vec4 binding
                obs.extend([lin_vel.x, lin_vel.y, lin_vel.z])
                obs.extend([ang_vel.x, ang_vel.y, ang_vel.z])
            else:
                # Pad with zeros if link missing
                obs.extend([0] * 13)
                
        return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        # The engine handles rendering
        pass
