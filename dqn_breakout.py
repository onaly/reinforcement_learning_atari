# -*- coding: utf-8 -*-

import math
import random
import warnings
from collections import namedtuple
from itertools import count

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

warnings.filterwarnings(action='ignore')

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, num_frames, h, w, num_outputs):
        super(DQN, self).__init__()
        # VOTRE CODE
        ############
        # Définition du réseau. Exemple :
        # 3 couches de convolution chacune suivie d'une batch normalization
        # filtres de taille 5 pixels, pas de 2
        # 16 filtres pour la première couche
        # 32 filtres pour la deuxième
        # 64 pour la troisième
        # Finir par une couche fully connected
        self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            """Fonction pour calculer la taille de sortie d'une couche convolutive"""
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, num_outputs)

    def forward(self, x):
        # VOTRE CODE
        ############
        # Calcul de la passe avant :
        # Fonction d'activation relu pour les couches cachées
        # Fonction d'activation linéaire sur la couche de sortie
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Agent:
    def __init__(self,env):

        self.env = env #initialisation de l'environnement
        self.batch_size = 32 #taille de batch
        self.gamma = 0.99 #coefficient gamma utilisé dans le calcul du gain
        self.eps_start = 0.9 #epsilon au début
        self.eps_end = 0.05 #epsilon à la fin
        self.eps_decay = 200 #taux de décroissance d'epsilon
        self.target_update = 10 #pas de mise à jour du réseau cible
        self.num_episodes = 1000 #nombre d'épisodes d'entraînement
        self.num_frames = 4 #nombre d'images dans une séquence

        self.im_height = 84 #hauteur d'une image
        self.im_width = 84 #largeur d'une image

        self.n_actions = env.action_space.n #nombre d'actions possibles

        self.episode_durations = [] #liste pour stocker les durées des épisodes

        #initialisation du réseau apprenant la fonction q :
        self.policy_net = DQN(self.num_frames, self.im_height, self.im_width, self.n_actions).to(device)
        #initialisation du réseau cible (pour le calcul des sorties cibles) :
        self.target_net = DQN(self.num_frames, self.im_height, self.im_width, self.n_actions).to(device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()

        #initialisation de l'optimiseur :
        self.optimizer = optim.Adam(self.policy_net.parameters(),1e-4)
        self.memory = ReplayMemory(10000) #initialisation du Replay Memory

        self.steps_done = 0 #nombre de pas réalisé au total

    def select_action(self,state,train=True):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if train:
            if sample > eps_threshold:
                # VOTRE CODE
                ############
                # Calcul et renvoi de l'action fournie par le réseau
                with torch.no_grad():
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                # VOTRE CODE
                ############
                # Calcul et renvoi d'une action choisie aléatoirement
                return torch.tensor(
                    [[random.randrange(self.n_actions)]], 
                    device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def process(self, state):
        """Fonction qui permet de transformer l'image d'un état en niveaux de gris
        puis on la redimensionne selon les tailles définies
        et on crée un tenseur Pytorch"""
        state = resize(rgb2gray(state), (self.im_height, self.im_width), mode='reflect') * 255
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=device, dtype=torch.float)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        # Masque des états non finaux :
        # Les états finaux sont ceux pour qui l'épisode se termine
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        # Calcul des états non finaux :
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # VOTRE CODE
        ############
        # Calcul de Q(s_t,a) : Q pour l'état courant
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # VOTRE CODE
        ############
        # Calcul de Q pour l'état suivant
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # VOTRE CODE
        ############
        # Calcul de Q future attendue cumulée
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # VOTRE CODE
        ############
        # Calcul de la fonction de perte de Huber
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # VOTRE CODE
        ############
        # Optimisation du modèle
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_policy_model(self):

        for i_episode in range(self.num_episodes):

            state = self.env.reset()
            state = self.process(state)

            for t in count():

                while state.size()[1] < self.num_frames:
                    action = 1 # Fire

                    new_frame, reward, done, _ = self.env.step(action)
                    new_frame = self.process(new_frame)

                    state = torch.cat([state, new_frame], 1)

                action = self.select_action(state)
                new_frame, reward, done, _ = self.env.step(action.item())
                new_frame = self.process(new_frame)

                if done:
                    new_state = None
                else :
                    new_state = torch.cat([state, new_frame], 1)
                    new_state = new_state[:, 1:, :, :]

                reward = torch.tensor([reward], device=device, dtype=torch.float)

                self.memory.push(state, action, new_state, reward)

                state = new_state

                self.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()

        self.save_model()
        print('Training completed')
        plt.show()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./dqn_model")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./dqn_model", map_location=device))

    def test(self):
        print('Testing model:')
        for i_episode in range(self.num_episodes):
            print('episode: {}'.format(i_episode))

            state = self.env.reset()
            state = self.process(state)

            while True:
                # -------------------------------------------
                # Cette partie ne fonctionne que lorsque le code 
                # est lancé depuis un notebook :
                plt.imshow(self.env.render(mode='rgb_array'),
                        interpolation='none')
                display.clear_output(wait=True)
                display.display(plt.gcf())
                # ----------------------------------------------

                # VOTRE CODE
                ############
                # Sélection d'une action appliquée à l'environnement
                # et mise à jour de l'état

                # On attend qu'on ait assez d'images pour le servir au réseau :
                while state.size()[1] < self.num_frames:
                    action = 1 # Fire

                    new_frame, _, done, _ = self.env.step(action)
                    new_frame = self.process(new_frame)

                    state = torch.cat([state, new_frame], 1)

                # Dès qu'on a assez d'images, on choisit l'action 
                # (via exploitation seulement) :
                action = self.select_action(state,train=False)
                # On observe le nouvel état (nouvelle image) :
                new_frame, _, done, _ = self.env.step(action.item())
                # On la transforme :
                new_frame = self.process(new_frame)

                if done: # si l'épisode est terminé
                    new_state = None
                else : 
                    # sinon on ajoute la nouvelle image
                    # à l'état
                    new_state = torch.cat([state, new_frame], 1)
                    # et on enlève la première image de l'état
                    # pour qu'on garde toujours le même nombre d'images
                    new_state = new_state[:, 1:, :, :]

                state = new_state

                if done:
                    break

        print('Testing completed')

if __name__ == '__main__':

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    env = gym.make('BreakoutDeterministic-v4').unwrapped
    env.reset()

    agent = Agent(env)

    # Training phase
    agent.train_policy_model()

    #Testing phase
    agent.load_model()
    agent.test()

    env.close()
