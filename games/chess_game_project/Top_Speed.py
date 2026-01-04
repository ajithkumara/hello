from ursina import *
import random

app = Ursina()
window.title = "Top Speed 3D"
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False

# --- Assets & Settings ---
ground_color = color.gray
grass_color = color.green
sky_color = color.cyan

# --- Camera ---
camera.orthographic = False
camera.fov = 80

# --- Environment ---
ground = Entity(
    model='plane',
    scale=(100, 1, 100),
    color=grass_color,
    texture='white_cube',
    texture_scale=(100, 100),
    collider='box'
)

track_road = Entity(
    model='plane',
    scale=(100, 1, 10),
    color=color.black,
    position=(0, 0.01, 0),
    collider='box'
)

# Start/Finish Line
finish_line = Entity(
    model='cube',
    scale=(0.5, 0.02, 10),
    color=color.white,
    position=(40, 0.02, 0),
    collider='box'
)

walls = []
def create_wall(position, scale):
    walls.append(Entity(
        model='cube',
        color=color.orange,
        position=position,
        scale=scale,
        collider='box'
    ))

# Simple Track Walls
create_wall((0, 0.5, 5.5), (100, 1, 1))
create_wall((0, 0.5, -5.5), (100, 1, 1))
create_wall((-50, 0.5, 0), (1, 1, 12)) 
create_wall((50, 0.5, 0), (1, 1, 12)) 

# --- Car Controller ---
class Car(Entity):
    def __init__(self):
        super().__init__()
        self.model = 'quad'
        self.texture = 'car_texture_gaming_trans.png'
        self.color = color.white 
        self.scale = (2, 4, 1) # Adjusted for top-down sprite
        self.position = (-40, 0.1, 0)
        self.collider = 'box'
        self.rotation_x = 90 # Lay flat on ground
        
        self.speed = 0
        self.rotation_speed = 100
        self.max_speed = 20
        self.friction = 0.5
        self.acceleration = 10
        
        # No wheels needed for 2D sprite
        self.wheels = []

    def update(self):
        # Movement
        move_amount = self.speed * time.dt
        
        # Basic Physics/Collision check (Forward raycast)
        hit_info = raycast(self.position + Vec3(0, 0.5, 0), self.forward, distance=1.5, ignore=(self,))
        
        if held_keys['w']:
             self.speed += self.acceleration * time.dt
        elif held_keys['s']:
             self.speed -= self.acceleration * time.dt
        else:
             # Friction
             if self.speed > 0: self.speed -= self.friction * time.dt
             if self.speed < 0: self.speed += self.friction * time.dt
             
        # Max Speed Clamp
        self.speed = clamp(self.speed, -self.max_speed/2, self.max_speed)
        
        # Turning
        if self.speed != 0:
            turn_direction = 1 if self.speed > 0 else -1
            if held_keys['a']:
                self.rotation_y -= self.rotation_speed * time.dt * turn_direction
            if held_keys['d']:
                self.rotation_y += self.rotation_speed * time.dt * turn_direction
                
        # Apply Movement
        if not hit_info.hit:
            self.position += self.forward * move_amount
        else:
            self.speed = 0 # Stop if hitting wall

        # Camera Follow (Fixed Angle)
        camera.position = self.position + Vec3(0, 30, -30)
        camera.rotation = (45, 0, 0)

# --- Game Logic ---
player = Car()

# Sky
Sky()

# Instructions
Text(text="CONTROLS: W,A,S,D to Drive", position=(-0.85, 0.45), origin=(0, 0), scale=2)

app.run()
