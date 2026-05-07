import pygame
from config import *

class Button:
    def __init__(self, x, y, width, height, text, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.hovered = False
        
    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        
    def render(self, screen):
        color = COLORS['button_hover'] if self.hovered else COLORS['button_bg']
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, COLORS['text'], self.rect, 2, border_radius=10)
        
        font = pygame.font.Font(FONT_NAME, FONT_SIZE_SMALL)
        text_surface = font.render(self.text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_click(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.callback()
            return True
        return False

class UIManager:
    def __init__(self, screen, game_callback):
        self.screen = screen
        self.game_callback = game_callback
        self.font_large = pygame.font.Font(FONT_NAME, FONT_SIZE_LARGE)
        self.font_medium = pygame.font.Font(FONT_NAME, FONT_SIZE_MEDIUM)
        self.font_small = pygame.font.Font(FONT_NAME, FONT_SIZE_SMALL)
        self.font_tiny = pygame.font.Font(FONT_NAME, FONT_SIZE_TINY)
        
        self.mode_buttons = []
        self.menu_buttons = []
        
        self._setup_buttons()
    
    def _setup_buttons(self):
        button_width = 300
        button_height = 60
        start_x = (SCREEN_WIDTH - button_width) // 2
        
        modes = [
            ('classic', '经典模式'),
            ('timed', '计时模式'),
            ('obstacle', '障碍模式'),
            ('survival', '生存模式')
        ]
        
        self.mode_buttons = []
        for i, (mode_key, mode_name) in enumerate(modes):
            y = 250 + i * 80
            btn = Button(start_x, y, button_width, button_height, 
                        mode_name, lambda m=mode_key: None)
            self.mode_buttons.append((btn, mode_key))
        
        self.menu_buttons = [
            Button(start_x, 600, button_width, 50, '排行榜', lambda: None),
            Button(start_x, 660, button_width, 50, '退出游戏', lambda: None)
        ]
    
    def show_main_menu(self):
        self._setup_buttons()
        selected_mode = None
        
        while True:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for btn, mode_key in self.mode_buttons:
                        if btn.handle_click(mouse_pos):
                            return self._show_difficulty_select(mode_key)
                    
                    for i, btn in enumerate(self.menu_buttons):
                        if btn.handle_click(mouse_pos):
                            if i == 0:
                                return 'leaderboard'
                            elif i == 1:
                                return 'quit'
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'quit'
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                        mode_index = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4].index(event.key)
                        return self._show_difficulty_select(self.mode_buttons[mode_index][1])
                    elif event.key == pygame.K_5:
                        return 'leaderboard'
            
            for btn, _ in self.mode_buttons:
                btn.update(mouse_pos)
            
            for btn in self.menu_buttons:
                btn.update(mouse_pos)
            
            self._render_main_menu()
            pygame.display.flip()
    
    def _show_difficulty_select(self, mode):
        button_width = 250
        button_height = 50
        start_x = (SCREEN_WIDTH - button_width) // 2
        
        difficulties = [
            ('easy', '简单 (15x15)'),
            ('medium', '中等 (25x25)'),
            ('hard', '困难 (35x35)'),
            ('nightmare', '噩梦 (45x45)')
        ]
        
        diff_buttons = []
        for i, (diff_key, diff_name) in enumerate(difficulties):
            y = 300 + i * 60
            btn = Button(start_x, y, button_width, button_height, diff_name, None)
            diff_buttons.append((btn, diff_key))
        
        while True:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for btn, diff_key in diff_buttons:
                        if btn.handle_click(mouse_pos):
                            self.game_callback(mode, diff_key)
                            return 'playing'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'menu'
            
            for btn, _ in diff_buttons:
                btn.update(mouse_pos)
            
            self._render_difficulty_select(mode, diff_buttons)
            pygame.display.flip()
    
    def _render_main_menu(self):
        self.screen.fill(COLORS['background'])
        
        title = self.font_large.render('迷宫逃脱', True, COLORS['celebration'])
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, 100))
        self.screen.blit(title, title_rect)
        
        subtitle = self.font_small.render('Maze Escape', True, COLORS['text_secondary'])
        subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH//2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        mode_label = self.font_medium.render('选择游戏模式:', True, COLORS['text'])
        mode_rect = mode_label.get_rect(center=(SCREEN_WIDTH//2, 200))
        self.screen.blit(mode_label, mode_rect)
        
        for btn, mode_key in self.mode_buttons:
            btn.render(self.screen)
            
            desc = GAME_MODES[mode_key]['description']
            desc_text = self.font_tiny.render(desc, True, COLORS['text_secondary'])
            desc_rect = desc_text.get_rect(midtop=(btn.rect.centerx, btn.rect.bottom + 5))
            self.screen.blit(desc_text, desc_rect)
        
        for i, btn in enumerate(self.menu_buttons):
            btn.render(self.screen)
        
        controls_text = self.font_tiny.render(
            '控制: 方向键/WASD 移动 | ESC 暂停/返回 | 数字键快速选择', 
            True, COLORS['text_secondary'])
        controls_rect = controls_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 50))
        self.screen.blit(controls_text, controls_rect)
    
    def _render_difficulty_select(self, mode, diff_buttons):
        self.screen.fill(COLORS['background'])
        
        mode_name = GAME_MODES[mode]['name']
        title = self.font_large.render(f'{mode_name} - 选择难度', True, COLORS['text'])
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, 150))
        self.screen.blit(title, title_rect)
        
        for btn, _ in diff_buttons:
            btn.render(self.screen)
        
        hint = self.font_small.render('按 ESC 返回', True, COLORS['text_secondary'])
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 100))
        self.screen.blit(hint, hint_rect)
    
    def show_leaderboard(self, score_manager):
        entries = score_manager.get_top_scores(10)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 'quit'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return 'menu'
            
            self._render_leaderboard(entries)
            pygame.display.flip()
    
    def _render_leaderboard(self, entries):
        self.screen.fill(COLORS['background'])
        
        title = self.font_large.render('排行榜', True, COLORS['celebration'])
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, 80))
        self.screen.blit(title, title_rect)
        
        header_y = 150
        headers = ['排名', '分数', '模式', '关卡']
        header_widths = [100, 150, 200, 150]
        start_x = (SCREEN_WIDTH - sum(header_widths)) // 2
        
        for i, (header, width) in enumerate(zip(headers, header_widths)):
            x = start_x + sum(header_widths[:i])
            header_text = self.font_medium.render(header, True, COLORS['text'])
            self.screen.blit(header_text, (x + (width - header_text.get_width())//2, header_y))
        
        pygame.draw.line(self.screen, COLORS['text_secondary'], 
                        (start_x, header_y + 35), 
                        (start_x + sum(header_widths), header_y + 35), 2)
        
        row_height = 50
        for rank, entry in enumerate(entries, 1):
            y = header_y + 50 + (rank - 1) * row_height
            
            if rank <= 3:
                colors = [COLORS['celebration'], (192, 192, 192), (205, 127, 50)]
                rank_color = colors[rank - 1]
            else:
                rank_color = COLORS['text']
            
            rank_text = self.font_medium.render(f'#{rank}', True, rank_color)
            score_text = self.font_medium.render(str(entry['score']), True, COLORS['text'])
            mode_text = self.font_medium.render(GAME_MODES[entry['mode']]['name'], True, COLORS['text_secondary'])
            level_text = self.font_medium.render(str(entry['level']), True, COLORS['text_secondary'])
            
            values = [rank_text, score_text, mode_text, level_text]
            for i, (val, width) in enumerate(zip(values, header_widths)):
                x = start_x + sum(header_widths[:i])
                self.screen.blit(val, (x + (width - val.get_width())//2, y))
        
        if not entries:
            no_data = self.font_medium.render('暂无记录', True, COLORS['text_secondary'])
            no_data_rect = no_data.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(no_data, no_data_rect)
        
        hint = self.font_small.render('按 ESC 返回主菜单', True, COLORS['text_secondary'])
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 80))
        self.screen.blit(hint, hint_rect)
