

import pygame
from PIL import Image, ImageDraw
# === CONSTANTS === (UPPER_CASE names)
text_color = (255, 255, 179)
chosed_color = (255, 255, 204)
button_color = (255, 204, 153)

zeor_color =  (0,0,0,0)
#zeor_color.convert_alpha()
#zeor_color = Image.new(mode='RGBA', size=(2000, 1200))


SCREEN_WIDTH = 2000
SCREEN_HEIGHT = 1200


# === CLASSES === (CamelCase names)

class Button():

    def __init__(self, text,value, x=0, y=0, width=100, height=50, command=None):

        self.text = text
        self.value = value
        self.command = command

        self.image_normal = pygame.Surface((width, height))
        self.image_normal.fill(button_color)

        self.image_hovered = pygame.Surface((width, height))
        self.image_hovered.fill(chosed_color)

        self.image = self.image_normal
        self.rect = self.image.get_rect()

        font = pygame.font.Font('freesansbold.ttf', 15)

        text_image1 = font.render(text, True, text_color)
        text_rect1 = text_image1.get_rect(center=self.rect.center)
        text_image2 = font.render(text, True, button_color)
        text_rect2 = text_image2.get_rect(center=self.rect.center)

        self.image_normal.blit(text_image1, text_rect1)
        self.image_hovered.blit(text_image2, text_rect2)

        # you can't use it before `blit`
        self.rect.topleft = (x, y)

        self.hovered = False
        # self.clicked = False

    def update(self):

        if self.hovered:
            self.image = self.image_hovered
        else:
            self.image = self.image_normal

    def draw(self, surface):

        surface.blit(self.image, self.rect)

    def handle_event(self, event):

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                print('Clicked:', self.value)
                return self.value

                #
                # if self.command:
                #     self.command()


# === FUNCTIONS === (lower_case names)

# empty

# === MAIN === (lower_case names)

def main():
    # --- init ---

    pygame.init()
    image = pygame.Surface([2000, 1200], pygame.SRCALPHA, 32)
    image.set_alpha(0)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen_rect = screen.get_rect()

    clock = pygame.time.Clock()
    is_running = False

    btn1 = Button('KNIGHT',2, 725, 500, 100, 50)
    btn2 = Button('BISHOP',3, 875, 500, 100, 50)
    btn3 = Button('ROOK',4, 1025, 500, 100, 50)
    btn4 = Button('QUEEN',5, 1175, 500, 100, 50)

    # --- mainloop --- (don't change it)

    is_running = True

    while is_running:

        # --- events ---

        for event in pygame.event.get():

            # --- global events ---

            if event.type == pygame.QUIT:
                is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    is_running = False

            # --- objects events ---

            knight = btn1.handle_event(event)
            bishop = btn2.handle_event(event)
            rook = btn3.handle_event(event)
            queen = btn4.handle_event(event)
            if knight != None:
                return knight
            if bishop != None:
                return bishop
            if rook != None:
                return  rook
            if queen != None:
                return queen

        # --- updates ---

        btn1.update()
        btn2.update()
        btn3.update()
        btn4.update()

        # --- draws ---
        screen.fill(chosed_color)
        btn1.draw(screen)
        btn2.draw(screen)
        btn3.draw(screen)
        btn4.draw(screen)

        pygame.display.update()

        # --- FPS ---

        clock.tick(25)

    # --- the end ---

    pygame.quit()


# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()
