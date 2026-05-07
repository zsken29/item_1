import pygame
import random
from config import COLORS, CELL_SIZE, OBSTACLE_SPEED

class Obstacle:
    def __init__(self, x, y, maze):
        self.x = x
        self.y = y
        self.maze = maze
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.move_timer = 0
        self.move_interval = 500
        
    def update(self, current_time, obstacles):
        if current_time - self.move_timer > self.move_interval:
            self._try_move()
            self.move_timer = current_time
    
    def _try_move(self):
        dx, dy = self.direction
        
        if random.random() < 0.2:
            perpendicular = [(-dy, dx), (dy, -dx)]
            self.direction = random.choice(perpendicular)
            dx, dy = self.direction
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        if not self.maze.is_wall(new_x, new_y):
            occupied = any(o != self and o.x == new_x and o.y == new_y for o in obstacles)
            if not occupied:
                self.x = new_x
                self.y = new_y
            else:
                self.direction = (-dx, -dy)
        else:
            self.direction = (-dx, -dy)
    
    def check_collision(self, player_pos):
        player_y, player_x = player_pos
        return self.x == player_x and self.y == player_y
    
    def render(self, screen, offset_x=0, offset_y=0):
        center_x = self.x * CELL_SIZE + CELL_SIZE // 2 - offset_x
        center_y = self.y * CELL_SIZE + CELL_SIZE // 2 - offset_y
        
        pulse = abs(pygame.math.Vector2(1, 0).rotate(
            (pygame.time.get_ticks() // 100) % 360).x) * 2
        
        pygame.draw.circle(screen, COLORS['obstacle'], 
                          (int(center_x), int(center_y)), 
                          CELL_SIZE // 2 - 2 + int(pulse))
        
        pygame.draw.circle(screen, (255, 100, 100), 
                          (int(center_x), int(center_y)), 
                          CELL_SIZE // 4)
        
    def get_position(self):
        return (self.y, self.x)

class ObstacleManager:
    def __init__(self, maze):
        self.maze = maze
        self.obstacles = []
        
    def spawn_obstacles(self, count):
        self.obstacles = []
        path_cells = self.maze.get_path_cells()
        
        start = self.maze.get_start()
        exit_pos = self.maze.get_exit()
        
        available_cells = [c for c in path_cells if c != start and c != exit_pos]
        
        for _ in range(min(count, len(available_cells))):
            cell = random.choice(available_cells)
            available_cells.remove(cell)
            self.obstacles.append(Obstacle(cell[0], cell[1], self.maze))
    
    def update(self, current_time):
        for obstacle in self.obstacles:
            obstacle.update(current_time, self.obstacles)
    
    def check_collision(self, player_pos):
        for obstacle in self.obstacles:
            if obstacle.check_collision(player_pos):
                return obstacle
        return None
    
    def render(self, screen, offset_x=0, offset_y=0):
        for obstacle in self.obstacles:
            obstacle.render(screen, offset_x, offset_y)
