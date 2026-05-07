import random
import numpy as np
from config import MAZE_SIZES

class MazeGenerator:
    def __init__(self, difficulty='medium'):
        self.difficulty = difficulty
        self.size = MAZE_SIZES.get(difficulty, MAZE_SIZES['medium'])
        self.maze = None
        
    def generate(self):
        self.maze = np.ones((self.size * 2 + 1, self.size * 2 + 1), dtype=np.int8)
        self._carve(0, 0)
        self.maze[0, 0] = 0
        self.maze[self.size * 2, self.size * 2] = 0
        return self.maze
    
    def _carve(self, x, y):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.maze[ny * 2 + 1][nx * 2 + 1] == 1:
                self.maze[ny * 2 + 1][nx * 2 + 1] = 0
                self.maze[y * 2 + 1 + dy][x * 2 + 1 + dx] = 0
                self.maze[ny * 2 + 1 + dy][nx * 2 + 1 + dx] = 0
                self._carve(nx, ny)
    
    def get_start(self):
        return (0, 0)
    
    def get_exit(self):
        return (self.size * 2, self.size * 2)
    
    def get_path_cells(self):
        cells = []
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] == 0:
                    cells.append((x, y))
        return cells
    
    def is_wall(self, x, y):
        if 0 <= x < len(self.maze[0]) and 0 <= y < len(self.maze):
            return self.maze[y][x] == 1
        return True
    
    def get_valid_moves(self, x, y):
        moves = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not self.is_wall(nx, ny):
                moves.append((nx, ny))
        return moves
    
    def find_path(self, start, end):
        if self.is_wall(end[0], end[1]):
            return []
        
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == end:
                return path
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if not self.is_wall(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return []
