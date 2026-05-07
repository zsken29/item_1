import pygame
import sys
from config import *
from game import Game

class MazeGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('迷宫逃脱 - Maze Escape')
        
        self.clock = pygame.time.Clock()
        self.game = None
        self.state = 'menu'
        
    def start_game_with_mode(self, mode, difficulty):
        self.game = Game(self.screen)
        self.game.start_game(mode)
        self.game.difficulty = difficulty
        self.game._setup_level()
        self.state = 'playing'
        
    def run(self):
        while True:
            self.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == 'playing' and self.game:
                            if self.game.paused:
                                self.game.paused = False
                            else:
                                self.state = 'menu'
                        else:
                            pygame.quit()
                            sys.exit()
            
            if self.state == 'menu':
                from ui import UIManager
                ui = UIManager(self.screen, self.start_game_with_mode)
                result = ui.show_main_menu()
                
                if result == 'leaderboard':
                    from ui import UIManager
                    from score import ScoreManager
                    score_manager = ScoreManager()
                    ui = UIManager(self.screen, self.start_game_with_mode)
                    ui.show_leaderboard(score_manager)
                elif result == 'quit':
                    pygame.quit()
                    sys.exit()
            
            elif self.state == 'playing' and self.game:
                result = self.game.run()
                if result == 'menu':
                    self.state = 'menu'
                elif result == 'quit':
                    pygame.quit()
                    sys.exit()
            
            pygame.display.flip()

if __name__ == '__main__':
    game = MazeGame()
    game.run()
