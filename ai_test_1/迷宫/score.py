import json
import os
from config import LEADERBOARD_FILE, MAX_LEADERBOARD_ENTRIES

class ScoreManager:
    def __init__(self):
        self.scores = []
        self._load_scores()
    
    def _load_scores(self):
        if os.path.exists(LEADERBOARD_FILE):
            try:
                with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
                    self.scores = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.scores = []
    
    def _save_scores(self):
        try:
            with open(LEADERBOARD_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.scores, f, ensure_ascii=False, indent=2)
        except IOError:
            pass
    
    def add_score(self, score, mode, level):
        entry = {
            'score': score,
            'mode': mode,
            'level': level
        }
        self.scores.append(entry)
        self.scores.sort(key=lambda x: x['score'], reverse=True)
        self.scores = self.scores[:MAX_LEADERBOARD_ENTRIES]
        self._save_scores()
    
    def get_top_scores(self, count=10):
        return self.scores[:count]
    
    def get_scores_by_mode(self, mode):
        return [s for s in self.scores if s['mode'] == mode]
    
    def clear_scores(self):
        self.scores = []
        self._save_scores()
