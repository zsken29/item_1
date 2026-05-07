import pygame
from config import COLORS, CELL_SIZE, PLAYER_MOVE_COOLDOWN, PLAYER_FAST_MOVE_COOLDOWN

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.steps = 0
        self.speed_boost = False
        self.speed_boost_end = 0
        self.has_shield = False
        self.has_map = False
        self.last_move_time = 0
        
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.steps = 0
        self.speed_boost = False
        self.has_shield = False
        self.has_map = False
        self.last_move_time = 0
        
    def move(self, dx, dy, maze, current_time):
        cooldown = PLAYER_FAST_MOVE_COOLDOWN if self.speed_boost else PLAYER_MOVE_COOLDOWN
        
        if current_time - self.last_move_time < cooldown:
            return False
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        if not maze.is_wall(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.steps += 1
            self.last_move_time = current_time
            return True
        
        return False
    
    def update(self, current_time):
        if self.speed_boost and current_time > self.speed_boost_end:
            self.speed_boost = False
    
    def apply_speed_boost(self, duration, current_time):
        self.speed_boost = True
        self.speed_boost_end = current_time + duration
    
    def activate_shield(self):
        self.has_shield = True
    
    def use_shield(self):
        if self.has_shield:
            self.has_shield = False
            return True
        return False
    
    def activate_map(self):
        self.has_map = True
    
    def teleport(self, maze):
        path_cells = maze.get_path_cells()
        if len(path_cells) > 1:
            current = (self.y, self.x)
            available = [c for c in path_cells if c != current]
            if available:
                new_pos = available[pygame.time.get_ticks() % len(available)]
                self.x = new_pos[1]
                self.y = new_pos[0]
                return True
        return False
    
    def render(self, screen, offset_x=0, offset_y=0):
        player_color = COLORS['player_fast'] if self.speed_boost else COLORS['player']
        
        center_x = self.x * CELL_SIZE + CELL_SIZE // 2 - offset_x
        center_y = self.y * CELL_SIZE + CELL_SIZE // 2 - offset_y
        
        pygame.draw.circle(screen, player_color, (int(center_x), int(center_y)), CELL_SIZE // 2 - 2)
        
        if self.has_shield:
            pygame.draw.circle(screen, COLORS['shield_active'], 
                             (int(center_x), int(center_y)), CELL_SIZE // 2 + 2, 2)
        
        inner_color = (max(0, player_color[0] - 50), 
                      max(0, player_color[1] - 50), 
                      max(0, player_color[2] - 50))
        pygame.draw.circle(screen, inner_color, 
                          (int(center_x - 3), int(center_y - 3)), 3)
        
    def get_position(self):
        return (self.y, self.x)
