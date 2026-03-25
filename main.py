import pygame
import sys
from game import Game2048

# ------------------------------------------------------------------ #
# Layout constants                                                     #
# ------------------------------------------------------------------ #

WINDOW_W      = 500
GRID_PAD      = 12
CELL_SIZE     = (WINDOW_W - GRID_PAD * 7) // 4
HEADER_H      = 100
WINDOW_H      = WINDOW_W + HEADER_H - GRID_PAD

# ------------------------------------------------------------------ #
# Colours — match the real 2048 palette                               #
# ------------------------------------------------------------------ #

BG_COLOR    = (250, 248, 239)
GRID_COLOR  = (187, 173, 160)
EMPTY_COLOR = (205, 192, 180)

TILE_COLORS = {
    0:    (205, 192, 180),
    2:    (238, 228, 218),
    4:    (237, 224, 200),
    8:    (242, 177, 121),
    16:   (245, 149,  99),
    32:   (246, 124,  95),
    64:   (246,  94,  59),
    128:  (237, 207, 114),
    256:  (237, 204,  97),
    512:  (237, 200,  80),
    1024: (237, 197,  63),
    2048: (237, 194,  46),
}

TEXT_DARK   = (119, 110, 101)
TEXT_LIGHT  = (249, 246, 242)

KEY_TO_DIR = {
    pygame.K_LEFT:  0,
    pygame.K_RIGHT: 1,
    pygame.K_UP:    2,
    pygame.K_DOWN:  3,
}

# Animation: fraction of completion added each frame (~7 frames at 60 fps)
ANIM_SPEED = 0.15


# ------------------------------------------------------------------ #
# Drawing helpers                                                      #
# ------------------------------------------------------------------ #

def cell_pos(r, c):
    """Top-left pixel coordinate of cell (r, c)."""
    x = GRID_PAD + GRID_PAD * (c + 1) + CELL_SIZE * c
    y = HEADER_H + GRID_PAD * (r + 1) + CELL_SIZE * r
    return x, y


def tile_color(value):
    return TILE_COLORS.get(value, (60, 58, 50))


def tile_text_color(value):
    return TEXT_DARK if value in (2, 4) else TEXT_LIGHT


def draw_tile(surface, value, x, y, font_large, font_small):
    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, tile_color(value), rect, border_radius=6)
    if value != 0:
        font = font_small if value >= 1024 else font_large
        text = font.render(str(value), True, tile_text_color(value))
        surface.blit(text, text.get_rect(center=rect.center))


def draw_grid_bg(surface):
    pygame.draw.rect(
        surface, GRID_COLOR,
        pygame.Rect(GRID_PAD, HEADER_H, WINDOW_W - GRID_PAD * 2, WINDOW_W - GRID_PAD * 2),
        border_radius=8,
    )
    for r in range(4):
        for c in range(4):
            x, y = cell_pos(r, c)
            pygame.draw.rect(surface, EMPTY_COLOR, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), border_radius=6)


def draw_header(surface, score, font_ui, font_sm):
    surface.blit(font_ui.render("2048", True, TEXT_DARK), (GRID_PAD, 18))

    box = pygame.Rect(WINDOW_W - 130, 14, 118, 50)
    pygame.draw.rect(surface, GRID_COLOR, box, border_radius=6)
    surface.blit(
        font_sm.render("SCORE", True, TEXT_LIGHT),
        font_sm.render("SCORE", True, TEXT_LIGHT).get_rect(centerx=box.centerx, top=box.top + 6),
    )
    surface.blit(
        font_sm.render(str(score), True, TEXT_LIGHT),
        font_sm.render(str(score), True, TEXT_LIGHT).get_rect(centerx=box.centerx, top=box.top + 26),
    )
    surface.blit(
        font_sm.render("R = new game", True, (170, 160, 150)),
        (GRID_PAD, HEADER_H - 20),
    )


def draw_game_over(surface, score, font_title, font_ui, font_sm):
    # Full opaque end screen
    surface.fill((250, 248, 239))

    cx = WINDOW_W // 2
    cy = WINDOW_H // 2

    # "Game End" title
    title = font_title.render("Game End", True, TEXT_DARK)
    surface.blit(title, title.get_rect(center=(cx, cy - 80)))

    # Divider line
    pygame.draw.line(surface, GRID_COLOR, (cx - 100, cy - 40), (cx + 100, cy - 40), 3)

    # Score label
    lbl = font_sm.render("FINAL SCORE", True, (150, 140, 130))
    surface.blit(lbl, lbl.get_rect(center=(cx, cy)))

    # Score value
    score_surf = font_ui.render(str(score), True, (242, 177, 121))
    surface.blit(score_surf, score_surf.get_rect(center=(cx, cy + 50)))

    # Restart hint
    hint = font_sm.render("Press R to play again", True, (170, 160, 150))
    surface.blit(hint, hint.get_rect(center=(cx, cy + 120)))


def ease_out(t):
    return 1 - (1 - t) ** 2


# ------------------------------------------------------------------ #
# Main loop                                                            #
# ------------------------------------------------------------------ #

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("2048")

    font_large = pygame.font.SysFont("Arial", 42, bold=True)
    font_small = pygame.font.SysFont("Arial", 30, bold=True)
    font_ui    = pygame.font.SysFont("Arial", 48, bold=True)
    font_sm    = pygame.font.SysFont("Arial", 20, bold=True)
    font_title = pygame.font.SysFont("Arial", 72, bold=True)

    game  = Game2048()
    clock = pygame.time.Clock()

    # Animation state
    # anim_t: 0.0 = animation just started, 1.0 = done (or idle)
    anim_t     = 1.0
    # anim_tiles: list of (px_from, py_from, px_to, py_to, value)
    anim_tiles = []

    # End-screen delay: timestamp (ms) when game_over became True, None otherwise
    game_over_at   = None
    GAME_OVER_DELAY = 2000   # ms before end screen appears

    while True:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    anim_t = 1.0
                    anim_tiles = []
                    game_over_at = None

                elif event.key in KEY_TO_DIR and not game.game_over and anim_t >= 1.0:
                    valid, tile_moves, _ = game.step(KEY_TO_DIR[event.key])
                    if valid:
                        anim_tiles = [
                            cell_pos(fr, fc) + cell_pos(tr, tc) + (val,)
                            for fr, fc, tr, tc, val in tile_moves
                        ]
                        anim_t = 0.0
                        if game.game_over and game_over_at is None:
                            game_over_at = now

        # Advance animation
        if anim_t < 1.0:
            anim_t = min(1.0, anim_t + ANIM_SPEED)

        t = ease_out(anim_t)

        # --- Render ---
        screen.fill(BG_COLOR)
        draw_header(screen, game.score, font_ui, font_sm)
        draw_grid_bg(screen)

        if anim_t < 1.0:
            # Draw each tile sliding from its old position to its new position
            for px0, py0, px1, py1, val in anim_tiles:
                x = int(px0 + (px1 - px0) * t)
                y = int(py0 + (py1 - py0) * t)
                draw_tile(screen, val, x, y, font_large, font_small)
        else:
            # Animation complete — draw the authoritative board
            for r in range(4):
                for c in range(4):
                    if game.board[r][c] != 0:
                        draw_tile(screen, game.board[r][c], *cell_pos(r, c), font_large, font_small)

        if game.game_over and game_over_at is not None and now - game_over_at >= GAME_OVER_DELAY:
            draw_game_over(screen, game.score, font_title, font_ui, font_sm)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
