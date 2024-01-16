import tensorflow as tf
import numpy as np
import pygame
import random
import os

pygame.font.init()

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 800
FONT = pygame.font.SysFont("comicsans", 50)
DRAW_LINES = True

WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Flappy Bird")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "bg.png")).convert_alpha(), (600, 900))
bird_imgs = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png"))) for x in
             range(1, 4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())

FLOOR_Y = 730
epoch = 0
max_score = 0

class QLearningAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.5  # exploration-exploitation trade-off
        self.epsilon_decay = 0.995  # decay rate for exploration
        self.epsilon_min = 0.001  # minimum exploration rate
        self.Q = {}  # Q-value table

    def act(self, state):
        state_key = tuple(state)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.Q.get(state_key, [0] * self.action_size))

    # def train(self, state, action, reward, next_state):
    #     state_key = tuple(state)
    #     next_state_key = tuple(next_state)
    #
    #     if state_key not in self.Q:
    #         self.Q[state_key] = [0] * self.action_size
    #
    #     if next_state_key not in self.Q:
    #         self.Q[next_state_key] = [0] * self.action_size
    #
    #     target = reward + self.gamma * np.max(self.Q[next_state_key])
    #     self.Q[state_key][action] = (1 - self.alpha) * self.Q[state_key][action] + self.alpha * target
    #
    #     # Update exploration-exploitation trade-off
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    def train(self, state, action, reward, next_state):
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        if state_key not in self.Q:
            self.Q[state_key] = [0] * self.action_size

        if next_state_key not in self.Q:
            self.Q[next_state_key] = [0] * self.action_size

        target = reward + self.gamma * np.max(self.Q[next_state_key])
        self.Q[state_key][action] = (1 - self.alpha) * self.Q[state_key][action] + self.alpha * target

        # Update exploration-exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.ticks = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = bird_imgs[0]

    def jump(self):
        self.vel = -10.5
        self.ticks = 0
        self.height = self.y

    def move(self):
        self.ticks += 1

        displacement = self.vel * (self.ticks) + 0.5 * 3 * (self.ticks) ** 2

        if displacement >= 16:
            displacement = (displacement / abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < 25:
                self.tilt = 25
        else:
            if self.tilt > -90:
                self.tilt -= 20

    def draw(self, win):
        self.img_count += 1

        if self.img_count <= 5:
            self.img = bird_imgs[0]
        elif self.img_count <= 10:
            self.img = bird_imgs[1]
        elif self.img_count <= 15:
            self.img = bird_imgs[2]
        elif self.img_count <= 20:
            self.img = bird_imgs[1]
        elif self.img_count == 21:
            self.img = bird_imgs[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = bird_imgs[1]
            self.img_count = 10

        rotate_center(win, self.img, (self.x, self.y), self.tilt)

    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()

    GAP = 200
    VELOCITY = 5

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False


class Base:
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def rotate_center(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)

def draw_window(win, birds, pipes, base, score, epoch, pipe_ind):
    if epoch == 0:
        epoch = 1
    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                                 (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2, pipes[pipe_ind].height),
                                 5)
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2), (
                                     pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2,
                                     pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird.draw(win)

    score_label = FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WINDOW_WIDTH - score_label.get_width() - 15, 10))

    score_label = FONT.render("Epochs: " + str(epoch - 1), 1, (255, 255, 255))
    win.blit(score_label, (10, 10))

    score_label = FONT.render("Max score: " + str(max_score), 1, (255, 255, 255))
    win.blit(score_label, (10, 50))

    pygame.display.update()


def get_state(bird, pipes):
    return np.array([bird.y, pipes[0].x, pipes[0].height]) if pipes else np.zeros(3)


def run_q_learning():
    state_size = 3
    action_size = 2

    agent = QLearningAgent(state_size, action_size)

    base = Base(FLOOR_Y)
    pipes = [Pipe(700)]
    birds = [Bird(230, 350)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        state = get_state(birds[0], pipes)
        action = agent.act(state)

        for x, bird in enumerate(birds):
            bird.move()

        base.move()

        pipes_to_remove = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()

            for bird in birds:
                if pipe.collide(bird, WINDOW):
                    birds.remove(bird)

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                pipes_to_remove.append(pipe)

            if not pipe.passed and pipe.x < birds[0].x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            if score > max_score:
                max_score = score
            pipes.append(Pipe(WINDOW_WIDTH))

        for r in pipes_to_remove:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR_Y or bird.y < -50:
                birds.remove(bird)

        next_state = get_state(birds[0], pipes)
        reward = 1 if pipes else 0

        train_q_learning(agent, state, action, reward, next_state)

        draw_window(WINDOW, birds, pipes, base, score, epoch, pipe_ind)

# def train_q_learning(agent, state, action, reward, next_state):
#     target = reward + 0.95 * np.max(agent.Q.get(tuple(next_state), [0] * agent.action_size))
#     agent.train(state, action, target)


def train_q_learning(agent, state, action, reward, next_state):
    state_key = tuple(state)
    next_state_key = tuple(next_state)

    if state_key not in agent.Q:
        agent.Q[state_key] = [0] * agent.action_size

    if next_state_key not in agent.Q:
        agent.Q[next_state_key] = [0] * agent.action_size

    target = reward + agent.gamma * np.max(agent.Q[next_state_key])
    agent.Q[state_key][action] = (1 - agent.alpha) * agent.Q[state_key][action] + agent.alpha * target

    # Update exploration-exploitation trade-off
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Target: {target}")


# def train_q_learning(agent, state, action, reward, next_state):
#     target = reward + 0.95 * np.max(agent.model.predict(next_state.reshape(1, -1)))
#     target_f = agent.model.predict(state.reshape(1, -1))
#     target_f[0][action] = target
#     agent.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
reward = 0
def eval_q_learning_agent(agent, win):
    global WINDOW, epoch, reward

    birds = [Bird(230, 350)]
    base = Base(FLOOR_Y)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pipe_ind = 0
        if len(birds) > 0 and len(pipes) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        if len(birds) > 0:
            state = get_state(birds[0], pipes) if pipes else np.zeros(3)
            action = agent.act(state)

            for x, bird in enumerate(birds):
                if action == 1:
                    bird.jump()
                bird.move()

            base.move()

            pipes_to_remove = []
            add_pipe = False
            for pipe in pipes:
                pipe.move()

                for bird in birds:
                    if pipe.collide(bird, WINDOW):
                        birds.remove(bird)

                if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                    pipes_to_remove.append(pipe)

                if not pipe.passed and len(birds) > 0 and pipe.x < birds[0].x:
                    pipe.passed = True
                    add_pipe = True

            if add_pipe:
                score += 1
                pipes.append(Pipe(WINDOW_WIDTH))

            for r in pipes_to_remove:
                pipes.remove(r)

            for bird in birds:
                if bird.y + bird.img.get_height() - 10 >= FLOOR_Y or bird.y < -50:
                    birds.remove(bird)

            if reward < 0:
                reward = 0
            reward += 1

            if len(birds) > 0 and pipes:
                next_state = get_state(birds[0], pipes)
            else:
                reward -= 1000
                epoch += 1
                # pygame.time.delay(1000)  # Delay after all birds have died
                birds = [Bird(230, 350)]
                pipes = [Pipe(700)]
                base = Base(FLOOR_Y)
                score = 0
                next_state = np.zeros(3)

            print(reward)

            agent.train(state, action, reward, next_state)

            draw_window(WINDOW, birds, pipes, base, score, epoch, pipe_ind)

    pygame.quit()

if __name__ == '__main__':
    # Initialize Q-learning agent and run evaluation
    action_size = 2
    agent = QLearningAgent(action_size)

    eval_q_learning_agent(agent, WINDOW)
