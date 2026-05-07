import pygame
import random
from config import *
from maze_generator import MazeGenerator
from maze_renderer import MazeRenderer
from player import Player
from items import ItemManager
from obstacles import ObstacleManager
from score import ScoreManager
from ui import UIManager

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.maze_generator = None
        self.maze_renderer = None
        self.player = None
        self.item_manager = None
        self.obstacle_manager = None
        self.score_manager = ScoreManager()
        self.ui_manager = UIManager(screen)
        
        self.current_mode = 'classic'
        self.current_level = 1
        self.difficulty = 'easy'
        
        self.start_time = 0
        self.time_limit = 0
        self.remaining_time = 0
        
        self.item_effect = None
        self.item_effect_timer = 0
        
        self.paused = False
        self.game_over = False
        self.game_won = False
        
    def start_game(self, mode):
        self.current_mode = mode
        self.current_level = 1
        self._setup_level()
        
    def next_level(self):
        self.current_level += 1
        difficulties = ['easy', 'medium', 'hard', 'nightmare']
        level_index = min(self.current_level - 1, len(difficulties) - 1)
        self.difficulty = difficulties[level_index]
        self._setup_level()
        
    def _setup_level(self):
        self.maze_generator = MazeGenerator(self.difficulty)
        self.maze = self.maze_generator.generate()
        self.maze_renderer = MazeRenderer(self.screen, self.maze)
        
        start = self.maze_generator.get_start()
        self.player = Player(start[1], start[0])
        
        self.item_manager = ItemManager(self.maze)
        item_count = 3 + self.current_level // 2
        self.item_manager.spawn_items(item_count)
        
        mode_config = GAME_MODES[self.current_mode]
        obstacle_count = mode_config['obstacles']
        if obstacle_count > 0:
            obstacle_count += self.current_level - 1
            self.obstacle_manager = ObstacleManager(self.maze)
            self.obstacle_manager.spawn_obstacles(obstacle_count)
        else:
            self.obstacle_manager = None
        
        if mode_config['time_limit']:
            self.time_limit = mode_config['time_limit'] * 1000
            if self.current_level > 1:
                self.time_limit = max(30000, self.time_limit - (self.current_level - 1) * 5000)
        else:
            self.time_limit = 0
        
        self.start_time = pygame.time.get_ticks()
        self.remaining_time = self.time_limit
        
        self.item_effect = None
        self.item_effect_timer = 0
        self.paused = False
        self.game_over = False
        self.game_won = False
        
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event.key)
            
            if not self.paused and not self.game_over and not self.game_won:
                self._update()
            
            self._render()
            pygame.display.flip()
            
            if self.game_won or self.game_over:
                result = self._show_result_screen()
                return result
        
        return 'menu'
    
    def _handle_keydown(self, key):
        if key == pygame.K_ESCAPE:
            if self.paused:
                self.paused = False
            else:
                self.paused = True
        elif key == pygame.K_r and (self.game_over or self.game_won):
            self._setup_level()
        elif key == pygame.K_RETURN and (self.game_over or self.game_won):
            if self.game_won:
                self.next_level()
            else:
                self._setup_level()
        
        if not self.paused and not self.game_over and not self.game_won:
            current_time = pygame.time.get_ticks()
            dx, dy = 0, 0
            
            if key in [pygame.K_UP, pygame.K_w]:
                dy = -1
            elif key in [pygame.K_DOWN, pygame.K_s]:
                dy = 1
            elif key in [pygame.K_LEFT, pygame.K_a]:
                dx = -1
            elif key in [pygame.K_RIGHT, pygame.K_d]:
                dx = 1
            
            if dx != 0 or dy != 0:
                self.player.move(dx, dy, self.maze, current_time)
    
    def _update(self):
        current_time = pygame.time.get_ticks()
        
        self.player.update(current_time)
        self.item_manager.update(current_time)
        
        if self.obstacle_manager:
            self.obstacle_manager.update(current_time)
        
        if self.item_effect:
            if current_time - self.item_effect_timer > 500:
                self.item_effect = None
        
        item = self.item_manager.check_collision(self.player.get_position())
        if item:
            self._apply_item_effect(item)
        
        exit_pos = self.maze_generator.get_exit()
        if self.player.x == exit_pos[0] and self.player.y == exit_pos[1]:
            self.game_won = True
            self.score = self._calculate_score()
            self.score_manager.add_score(self.score, self.current_mode, self.current_level)
        
        if self.time_limit > 0:
            self.remaining_time = self.time_limit - (current_time - self.start_time)
            if self.remaining_time <= 0:
                self.game_over = True
        
        if self.obstacle_manager:
            obstacle = self.obstacle_manager.check_collision(self.player.get_position())
            if obstacle:
                if self.player.use_shield():
                    pass
                else:
                    self.game_over = True
        
        if self.paused:
            return
    
    def _apply_item_effect(self, item):
        current_time = pygame.time.get_ticks()
        
        if item.type == 'speed':
            self.player.apply_speed_boost(ITEM_EFFECTS['speed']['duration'], current_time)
        elif item.type == 'shield':
            self.player.activate_shield()
        elif item.type == 'map':
            self.player.activate_map()
        elif item.type == 'teleport':
            self.player.teleport(self.maze)
        
        self.item_effect = item.type
        self.item_effect_timer = current_time
    
    def _calculate_score(self):
        base_score = SCORE_BASE * self.current_level
        time_bonus = 0
        if self.time_limit > 0:
            time_bonus = int((self.remaining_time / 1000) * SCORE_TIME_BONUS)
        
        step_penalty = self.player.steps * SCORE_STEP_PENALTY
        items_collected = sum(1 for i in self.item_manager.items if i.collected)
        item_bonus = items_collected * SCORE_ITEM_BONUS
        
        mode_config = GAME_MODES[self.current_mode]
        multiplier = mode_config['score_multiplier']
        
        total_score = int((base_score + time_bonus + item_bonus - step_penalty) * multiplier)
        return max(total_score, 100)
    
    def _render(self):
        self.maze_renderer.render(self.player.has_map)
        
        exit_pos = self.maze_generator.get_exit()
        self._render_exit(exit_pos)
        
        self.item_manager.render(self.screen, 
                                self.maze_renderer.offset_x, 
                                self.maze_renderer.offset_y)
        
        if self.obstacle_manager:
            self.obstacle_manager.render(self.screen,
                                        self.maze_renderer.offset_x,
                                        self.maze_renderer.offset_y)
        
        self.player.render(self.screen, 
                         self.maze_renderer.offset_x, 
                         self.maze_renderer.offset_y)
        
        if self.item_effect:
            self.item_manager.render_effect(self.screen, self.item_effect,
                                          self.player.x, self.player.y,
                                          self.maze_renderer.offset_x,
                                          self.maze_renderer.offset_y)
        
        self.maze_renderer.render_minimap(
            self.player.get_position(),
            (exit_pos[1], exit_pos[0]),
            self.item_manager.items,
            self.obstacle_manager.obstacles if self.obstacle_manager else []
        )
        
        self._render_hud()
        
        if self.paused:
            self._render_pause_screen()
    
    def _render_exit(self, exit_pos):
        center_x = exit_pos[0] * CELL_SIZE + CELL_SIZE // 2 - self.maze_renderer.offset_x
        center_y = exit_pos[1] * CELL_SIZE + CELL_SIZE // 2 - self.maze_renderer.offset_y
        
        pulse = abs(pygame.math.Vector2(1, 0).rotate(
            (pygame.time.get_ticks() // 200) % 360).x) * 3
        
        pygame.draw.rect(self.screen, COLORS['exit'],
                       (center_x - CELL_SIZE//2, center_y - CELL_SIZE//2,
                        CELL_SIZE, CELL_SIZE))
        pygame.draw.circle(self.screen, COLORS['celebration'],
                          (int(center_x), int(center_y)), 
                          CELL_SIZE//3 + int(pulse))
    
    def _render_hud(self):
        font = pygame.font.Font(FONT_NAME, FONT_SIZE_SMALL)
        
        mode_name = GAME_MODES[self.current_mode]['name']
        mode_text = font.render(f'模式: {mode_name}', True, COLORS['text'])
        self.screen.blit(mode_text, (20, 20))
        
        level_text = font.render(f'关卡: {self.current_level}', True, COLORS['text'])
        self.screen.blit(level_text, (20, 50))
        
        steps_text = font.render(f'步数: {self.player.steps}', True, COLORS['text'])
        self.screen.blit(steps_text, (20, 80))
        
        if self.player.has_map:
            map_indicator = font.render('地图', True, COLORS['item_map'])
            self.screen.blit(map_indicator, (20, 110))
        
        if self.time_limit > 0:
            time_seconds = max(0, self.remaining_time // 1000)
            time_text = font.render(f'剩余时间: {time_seconds}秒', True, COLORS['text'])
            self.screen.blit(time_text, (20, 140))
    
    def _render_pause_screen(self):
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        font = pygame.font.Font(FONT_NAME, FONT_SIZE_LARGE)
        text = font.render('游戏暂停', True, COLORS['text'])
        rect = text.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2 - 50))
        self.screen.blit(text, rect)
        
        font_small = pygame.font.Font(FONT_NAME, FONT_SIZE_MEDIUM)
        hint1 = font_small.render('按 ESC 继续游戏', True, COLORS['text_secondary'])
        hint2 = font_small.render('按 R 返回主菜单', True, COLORS['text_secondary'])
        self.screen.blit(hint1, hint1.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2 + 20)))
        self.screen.blit(hint2, hint2.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2 + 60)))
    
    def _show_result_screen(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'menu'
                    elif event.key == pygame.K_RETURN:
                        if self.game_won:
                            self.next_level()
                        else:
                            self._setup_level()
                        waiting = False
                    elif event.key == pygame.K_r:
                        self._setup_level()
                        waiting = False
            
            self._render_result_screen()
            pygame.display.flip()
        
        return 'playing'
    
    def _render_result_screen(self):
        overlay = pygame.Surface(self.screen.get_size())
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        font = pygame.font.Font(FONT_NAME, FONT_SIZE_LARGE)
        
        if self.game_won:
            title = '通关成功!'
            title_color = COLORS['celebration']
        else:
            title = '游戏结束'
            title_color = COLORS['obstacle']
        
        text = font.render(title, True, title_color)
        rect = text.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2 - 80))
        self.screen.blit(text, rect)
        
        if self.game_won:
            font_medium = pygame.font.Font(FONT_NAME, FONT_SIZE_MEDIUM)
            score_text = font_medium.render(f'得分: {self.score}', True, COLORS['text'])
            self.screen.blit(score_text, score_text.get_rect(
                center=(self.screen.get_width()//2, self.screen.get_height()//2 - 20)))
            
            stats_text = font_medium.render(f'关卡: {self.current_level} | 步数: {self.player.steps}', 
                                          True, COLORS['text_secondary'])
            self.screen.blit(stats_text, stats_text.get_rect(
                center=(self.screen.get_width()//2, self.screen.get_height()//2 + 20)))
            
            hint1 = font_medium.render('按 ENTER 进入下一关', True, COLORS['text'])
            self.screen.blit(hint1, hint1.get_rect(
                center=(self.screen.get_width()//2, self.screen.get_height()//2 + 80)))
        else:
            font_medium = pygame.font.Font(FONT_NAME, FONT_SIZE_MEDIUM)
            hint1 = font_medium.render('很遗憾，你失败了', True, COLORS['text_secondary'])
            self.screen.blit(hint1, hint1.get_rect(
                center=(self.screen.get_width()//2, self.screen.get_height()//2 - 20)))
        
        hint2 = font_medium.render('按 R 重新开始 | 按 ESC 返回菜单', True, COLORS['text_secondary'])
        self.screen.blit(hint2, hint2.get_rect(
            center=(self.screen.get_width()//2, self.screen.get_height()//2 + 130)))
    
    def show_menu(self):
        return self.ui_manager.show_main_menu()
    
    def show_leaderboard(self):
        return self.ui_manager.show_leaderboard(self.score_manager)
