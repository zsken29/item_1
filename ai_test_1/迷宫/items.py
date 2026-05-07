import pygame
import random
from config import COLORS, CELL_SIZE, ITEM_SPAWN_CHANCE

class Item:
    def __init__(self, x, y, item_type):
        self.x = x
        self.y = y
        self.type = item_type
        self.collected = False
        self.animation_offset = 0
        self.animation_time = 0
        
    def update(self, current_time):
        self.animation_time = current_time
        self.animation_offset = abs(pygame.math.Vector2(0, 1).rotate(
            (current_time // 100) % 360).y) * 3
    
    def render(self, screen, offset_x=0, offset_y=0):
        if self.collected:
            return
        
        center_x = self.x * CELL_SIZE + CELL_SIZE // 2 - offset_x
        center_y = self.y * CELL_SIZE + CELL_SIZE // 2 - offset_y - self.animation_offset
        
        color = COLORS.get(f'item_{self.type}', COLORS['text'])
        
        if self.type == 'speed':
            pygame.draw.polygon(screen, color, [
                (center_x, center_y - 6),
                (center_x + 6, center_y + 4),
                (center_x - 6, center_y + 4)
            ])
        elif self.type == 'shield':
            pygame.draw.circle(screen, color, (int(center_x), int(center_y)), 6, 2)
            pygame.draw.circle(screen, color, (int(center_x), int(center_y)), 3)
        elif self.type == 'map':
            pygame.draw.rect(screen, color, 
                           (center_x - 5, center_y - 5, 10, 10))
            pygame.draw.line(screen, COLORS['background'], 
                           (center_x - 3, center_y), (center_x + 3, center_y))
            pygame.draw.line(screen, COLORS['background'], 
                           (center_x, center_y - 3), (center_x, center_y + 3))
        elif self.type == 'teleport':
            points = []
            for i in range(6):
                angle = i * 60 + (self.animation_time // 200) % 360
                rad = pygame.math.Vector2(1, 0).rotate(angle)
                points.append((center_x + rad.x * 6, center_y + rad.y * 6))
            pygame.draw.polygon(screen, color, points)

class ItemManager:
    def __init__(self, maze):
        self.maze = maze
        self.items = []
        self.item_types = ['speed', 'shield', 'map', 'teleport']
        
    def spawn_items(self, count=5):
        self.items = []
        path_cells = self.maze.get_path_cells()
        
        start = self.maze.get_start()
        exit_pos = self.maze.get_exit()
        
        available_cells = [c for c in path_cells if c != start and c != exit_pos]
        
        spawn_count = min(count, len(available_cells))
        selected_cells = random.sample(available_cells, spawn_count)
        
        for cell in selected_cells:
            item_type = random.choice(self.item_types)
            self.items.append(Item(cell[0], cell[1], item_type))
    
    def check_collision(self, player_pos):
        player_y, player_x = player_pos
        for item in self.items:
            if not item.collected and item.x == player_x and item.y == player_y:
                item.collected = True
                return item
        return None
    
    def update(self, current_time):
        for item in self.items:
            item.update(current_time)
    
    def render(self, screen, offset_x=0, offset_y=0):
        for item in self.items:
            item.render(screen, offset_x, offset_y)
    
    def render_effect(self, screen, item_type, player_x, player_y, offset_x, offset_y):
        center_x = player_x * CELL_SIZE + CELL_SIZE // 2 - offset_x
        center_y = player_y * CELL_SIZE + CELL_SIZE // 2 - offset_y
        color = COLORS.get(f'item_{item_type}', COLORS['text'])
        
        time_factor = (pygame.time.get_ticks() % 500) / 500
        radius = 10 + time_factor * 20
        alpha = int(255 * (1 - time_factor))
        
        effect_surface = pygame.Surface((int(radius * 2 + 10), int(radius * 2 + 10)), pygame.SRCALPHA)
        pygame.draw.circle(effect_surface, (*color, alpha), 
                          (int(radius + 5), int(radius + 5)), int(radius), 2)
        screen.blit(effect_surface, (center_x - radius - 5, center_y - radius - 5))
