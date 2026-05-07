import pygame
import os

pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

MAZE_SIZES = {
    'easy': 15,
    'medium': 25,
    'hard': 35,
    'nightmare': 45
}

MAZE_DIFFICULTIES = ['easy', 'medium', 'hard', 'nightmare']

CELL_SIZE = 20

COLORS = {
    'background': (30, 30, 40),
    'wall': (60, 60, 80),
    'path': (45, 45, 55),
    'player': (100, 200, 100),
    'player_fast': (150, 255, 150),
    'exit': (255, 200, 100),
    'item_speed': (255, 100, 100),
    'item_shield': (100, 150, 255),
    'item_map': (255, 255, 100),
    'item_teleport': (200, 100, 255),
    'obstacle': (255, 50, 50),
    'text': (255, 255, 255),
    'text_secondary': (180, 180, 180),
    'button_bg': (50, 80, 120),
    'button_hover': (70, 100, 150),
    'shield_active': (100, 150, 255, 100),
    'celebration': (255, 215, 0)
}

FONT_NAME = None
FONT_SIZE_LARGE = 48
FONT_SIZE_MEDIUM = 32
FONT_SIZE_SMALL = 24
FONT_SIZE_TINY = 18

PLAYER_MOVE_COOLDOWN = 150
PLAYER_FAST_MOVE_COOLDOWN = 100

ITEM_EFFECTS = {
    'speed': {'duration': 5000, 'color': COLORS['item_speed']},
    'shield': {'uses': 1, 'color': COLORS['item_shield']},
    'map': {'permanent': True, 'color': COLORS['item_map']},
    'teleport': {'immediate': True, 'color': COLORS['item_teleport']}
}

GAME_MODES = {
    'classic': {
        'name': '经典模式',
        'description': '到达终点即可通关，无时间限制',
        'time_limit': None,
        'obstacles': 0,
        'score_multiplier': 1.0
    },
    'timed': {
        'name': '计时模式',
        'description': '在时间限制内到达终点',
        'time_limit': 120,
        'obstacles': 0,
        'score_multiplier': 1.5
    },
    'obstacle': {
        'name': '障碍模式',
        'description': '地图上有移动障碍物，触碰会扣分',
        'time_limit': None,
        'obstacles': 3,
        'score_multiplier': 2.0
    },
    'survival': {
        'name': '生存模式',
        'description': '有时间限制且有障碍物',
        'time_limit': 90,
        'obstacles': 5,
        'score_multiplier': 3.0
    }
}

OBSTACLE_SPEED = 1

ITEM_SPAWN_CHANCE = 0.15

SCORE_BASE = 1000
SCORE_TIME_BONUS = 10
SCORE_STEP_PENALTY = 1
SCORE_ITEM_BONUS = 50
SCORE_OBSTACLE_PENALTY = 100

LEADERBOARD_FILE = 'leaderboard.json'
MAX_LEADERBOARD_ENTRIES = 10

ASSETS_DIR = 'assets'
SOUND_DIR = os.path.join(ASSETS_DIR, 'sounds')
MUSIC_DIR = os.path.join(ASSETS_DIR, 'music')

SOUNDS = {
    'move': 'move.wav',
    'pickup': 'pickup.wav',
    'win': 'win.wav',
    'lose': 'lose.wav',
    'teleport': 'teleport.wav',
    'hit': 'hit.wav'
}

MUSIC = {
    'menu': 'menu_music.mp3',
    'game': 'game_music.mp3',
    'victory': 'victory.mp3'
}

pygame.quit()
