import pygame
import os
from config import SOUNDS, MUSIC, ASSETS_DIR

class AudioManager:
    def __init__(self):
        self.enabled = True
        self.sound_volume = 0.5
        self.music_volume = 0.3
        
        self.sounds = {}
        self.current_music = None
        
        self._load_sounds()
    
    def _load_sounds(self):
        for name, filename in SOUNDS.items():
            path = os.path.join(ASSETS_DIR, 'sounds', filename)
            if os.path.exists(path):
                try:
                    self.sounds[name] = pygame.mixer.Sound(path)
                    self.sounds[name].set_volume(self.sound_volume)
                except pygame.error:
                    pass
    
    def play_sound(self, name):
        if self.enabled and name in self.sounds:
            try:
                self.sounds[name].play()
            except pygame.error:
                pass
    
    def play_music(self, name, loops=-1):
        if not self.enabled:
            return
        
        path = os.path.join(ASSETS_DIR, 'music', MUSIC.get(name))
        if path and os.path.exists(path):
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(self.music_volume)
                pygame.mixer.music.play(loops)
                self.current_music = name
            except pygame.error:
                pass
    
    def stop_music(self):
        try:
            pygame.mixer.music.stop()
            self.current_music = None
        except pygame.error:
            pass
    
    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.stop_music()
        return self.enabled
    
    def set_sound_volume(self, volume):
        self.sound_volume = max(0.0, min(1.0, volume))
        for sound in self.sounds.values():
            sound.set_volume(self.sound_volume)
    
    def set_music_volume(self, volume):
        self.music_volume = max(0.0, min(1.0, volume))
        try:
            pygame.mixer.music.set_volume(self.music_volume)
        except pygame.error:
            pass
