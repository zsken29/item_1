import pygame
from config import COLORS, CELL_SIZE

class MazeRenderer:
    def __init__(self, screen, maze):
        self.screen = screen
        self.maze = maze
        self.offset_x = 0
        self.offset_y = 0
        
    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y
        
    def render(self, show_full_maze=False):
        maze_height = len(self.maze) * CELL_SIZE
        maze_width = len(self.maze[0]) * CELL_SIZE
        
        visible_width = self.screen.get_width() - 400
        visible_height = self.screen.get_height() - 200
        
        if maze_width > visible_width:
            max_offset_x = maze_width - visible_width
            self.offset_x = min(self.offset_x, max_offset_x)
            self.offset_x = max(0, self.offset_x)
        
        if maze_height > visible_height:
            max_offset_y = maze_height - visible_height
            self.offset_y = min(self.offset_y, max_offset_y)
            self.offset_y = max(0, self.offset_y)
        
        self._render_background()
        
        if show_full_maze:
            self._render_full_maze()
        else:
            self._render_visible_maze()
    
    def _render_background(self):
        self.screen.fill(COLORS['background'])
    
    def _render_full_maze(self):
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                rect = pygame.Rect(
                    x * CELL_SIZE - self.offset_x,
                    y * CELL_SIZE - self.offset_y,
                    CELL_SIZE,
                    CELL_SIZE
                )
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, COLORS['wall'], rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['path'], rect)
                    pygame.draw.rect(self.screen, COLORS['background'], rect.inflate(-2, -2))
    
    def _render_visible_maze(self):
        visible_cols = self.screen.get_width() // CELL_SIZE + 2
        visible_rows = self.screen.get_height() // CELL_SIZE + 2
        
        start_col = max(0, self.offset_x // CELL_SIZE)
        start_row = max(0, self.offset_y // CELL_SIZE)
        
        for y in range(start_row, min(len(self.maze), start_row + visible_rows)):
            for x in range(start_col, min(len(self.maze[0]), start_col + visible_cols)):
                screen_x = x * CELL_SIZE - self.offset_x
                screen_y = y * CELL_SIZE - self.offset_y
                
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, COLORS['wall'], 
                                   (screen_x, screen_y, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(self.screen, COLORS['path'], 
                                   (screen_x, screen_y, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(self.screen, COLORS['background'], 
                                   (screen_x + 1, screen_y + 1, CELL_SIZE - 2, CELL_SIZE - 2))
    
    def render_minimap(self, player_pos, exit_pos, items, obstacles):
        minimap_size = 150
        margin = 20
        minimap_x = self.screen.get_width() - minimap_size - margin
        minimap_y = margin
        
        minimap_surface = pygame.Surface((minimap_size, minimap_size))
        minimap_surface.set_alpha(200)
        minimap_surface.fill(COLORS['background'])
        
        maze_height = len(self.maze)
        maze_width = len(self.maze[0])
        cell_size = minimap_size / max(maze_width, maze_height)
        
        for y in range(min(maze_height, int(minimap_size / cell_size))):
            for x in range(min(maze_width, int(minimap_size / cell_size))):
                if y < maze_height and x < maze_width:
                    if self.maze[y][x] == 1:
                        pygame.draw.rect(minimap_surface, COLORS['wall'],
                                       (x * cell_size, y * cell_size, cell_size, cell_size))
        
        pygame.draw.rect(minimap_surface, COLORS['exit'],
                        (exit_pos[1] * cell_size, exit_pos[0] * cell_size, 
                         cell_size * 1.5, cell_size * 1.5))
        
        for item in items:
            item_color = COLORS.get(f'item_{item.type}', COLORS['text'])
            pygame.draw.circle(minimap_surface, item_color,
                              (int(item.x * cell_size + cell_size/2), 
                               int(item.y * cell_size + cell_size/2)),
                              cell_size)
        
        for obs in obstacles:
            pygame.draw.circle(minimap_surface, COLORS['obstacle'],
                              (int(obs.x * cell_size + cell_size/2),
                               int(obs.y * cell_size + cell_size/2)),
                              cell_size)
        
        pygame.draw.circle(minimap_surface, COLORS['player'],
                          (int(player_pos[1] * cell_size + cell_size/2),
                           int(player_pos[0] * cell_size + cell_size/2)),
                          cell_size)
        
        pygame.draw.rect(minimap_surface, COLORS['text'], 
                        minimap_surface.get_rect(), 2)
        
        self.screen.blit(minimap_surface, (minimap_x, minimap_y))
