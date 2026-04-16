import pygame
import sys
import numpy as np
from game import Game2048
from local_search import SimulatedAnnealingSolver

# Constants
WINDOW_W      = 500
GRID_PAD      = 12
CELL_SIZE     = (WINDOW_W - GRID_PAD * 7) // 4
HEADER_H      = 100
WINDOW_H      = WINDOW_W + HEADER_H - GRID_PAD
ANIM_SPEED    = 0.25  # Increased for faster AI play

# Colors
BG_COLOR    = (250, 248, 239)
GRID_COLOR  = (187, 173, 160)
EMPTY_COLOR = (205, 192, 180)
TEXT_DARK   = (119, 110, 101)
TEXT_LIGHT  = (249, 246, 242)

TILE_COLORS = {
    0: (205, 192, 180), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
}

# --- Drawing Helpers ---
def cell_pos(r, c):
    x = GRID_PAD + GRID_PAD * (c + 1) + CELL_SIZE * c
    y = HEADER_H + GRID_PAD * (r + 1) + CELL_SIZE * r
    return x, y

def draw_tile(surface, value, x, y, font_large, font_small):
    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    color = TILE_COLORS.get(value, (60, 58, 50))
    pygame.draw.rect(surface, color, rect, border_radius=6)
    if value != 0:
        font = font_small if value >= 1024 else font_large
        text_col = TEXT_DARK if value in (2, 4) else TEXT_LIGHT
        text = font.render(str(value), True, text_col)
        surface.blit(text, text.get_rect(center=rect.center))

def draw_grid_bg(surface):
    pygame.draw.rect(surface, GRID_COLOR, pygame.Rect(GRID_PAD, HEADER_H, WINDOW_W - GRID_PAD * 2, WINDOW_W - GRID_PAD * 2), border_radius=8)
    for r in range(4):
        for c in range(4):
            pygame.draw.rect(surface, EMPTY_COLOR, pygame.Rect(*cell_pos(r, c), CELL_SIZE, CELL_SIZE), border_radius=6)

def draw_header(surface, score, temp, font_ui, font_sm):
    surface.blit(font_ui.render("2048 AI", True, TEXT_DARK), (GRID_PAD, 18))
    # Score Box
    box = pygame.Rect(WINDOW_W - 130, 14, 118, 50)
    pygame.draw.rect(surface, GRID_COLOR, box, border_radius=6)
    surface.blit(font_sm.render("SCORE", True, TEXT_LIGHT), font_sm.render("SCORE", True, TEXT_LIGHT).get_rect(centerx=box.centerx, top=box.top + 6))
    surface.blit(font_sm.render(str(score), True, TEXT_LIGHT), font_sm.render(str(score), True, TEXT_LIGHT).get_rect(centerx=box.centerx, top=box.top + 26))
    # Temp Info
    temp_txt = f"Temp: {temp:.1f}"
    surface.blit(font_sm.render(temp_txt, True, TEXT_DARK), (GRID_PAD, HEADER_H - 25))

# --- Main AI Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("2048 - Simulated Annealing AI")

    fonts = {
        'lg': pygame.font.SysFont("Arial", 42, bold=True),
        'sm': pygame.font.SysFont("Arial", 30, bold=True),
        'ui': pygame.font.SysFont("Arial", 48, bold=True),
        'stat': pygame.font.SysFont("Arial", 18, bold=True)
    }

    game = Game2048()
    solver = SimulatedAnnealingSolver(initial_temp=2000.0, alpha=0.95)
    clock = pygame.time.Clock()

    anim_t, anim_tiles = 1.0, []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset(); solver = SimulatedAnnealingSolver(); anim_t = 1.0

        # AI Turn Logic
        if not game.game_over and anim_t >= 1.0:
            move = solver.get_best_move(game)
            if move is not None:
                valid, tile_moves, _ = game.step(move)
                if valid:
                    anim_tiles = [cell_pos(fr, fc) + cell_pos(tr, tc) + (val,) for fr, fc, tr, tc, val in tile_moves]
                    anim_t = 0.0

        # Animation logic
        if anim_t < 1.0: anim_t = min(1.0, anim_t + ANIM_SPEED)
        
        # Rendering
        screen.fill(BG_COLOR)
        draw_header(screen, game.score, solver.t, fonts['ui'], fonts['stat'])
        draw_grid_bg(screen)

        if anim_t < 1.0:
            t_ease = 1 - (1 - anim_t)**2
            for px0, py0, px1, py1, val in anim_tiles:
                draw_tile(screen, val, int(px0 + (px1 - px0) * t_ease), int(py0 + (py1 - py0) * t_ease), fonts['lg'], fonts['sm'])
        else:
            for r in range(4):
                for c in range(4):
                    if game.board[r][c] != 0:
                        draw_tile(screen, game.board[r][c], *cell_pos(r, c), fonts['lg'], fonts['sm'])

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()