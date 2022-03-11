# -*- coding: utf-8 -*-

import math
import random
import warnings
from collections import namedtuple

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

warnings.filterwarnings(action='ignore')

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):

    def __init__(self):
        super(QNet, self).__init__()
        # VOTRE CODE
        ############
        # Définition d'un réseau avec une couche cachée (à 256 neurones par exemple)
        self.l1 = nn.Linear(4, 164) # 4 entrées d'état
        self.l2 = nn.Linear(164, 2) # 2 sorties : 1 pour chaque action (gauche ou droite)

    def forward(self, x):
        # VOTRE CODE
        ############
        # Calcul de la passe avant :
        # Fonction d'activation relu pour la couche cachée
        # Fonction d'activation linéaire sur la couche de sortie
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self,env):

        self.env = env #l'environnement gym
        self.batch_size = 64 #taille de batch
        self.gamma = 0.99 #coefficient gamma utilisé dans le calcul du gain
        self.eps_start = 0.9 #epsilon au début
        self.eps_end = 0.05 #epsilon à la fin
        self.eps_decay = 200 #taux de décroissance d'epsilon
        self.target_update = 10 #pas de mise à jour du réseau cible
        self.num_episodes = 500 #nombre d'épisodes d'entraînement

        self.n_actions = env.action_space.n #nombre d'actions possibles
        self.episode_durations = [] #liste pour stocker les durées des épisodes
        
        self.policy_net = QNet() #initialisation du réseau apprenant la fonction q
        self.target_net = QNet() #initialisation du réseau cible (pour le calcul des sorties cibles)

        if use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters()) #initialisation de l'optimiseur
        self.memory = ReplayMemory(10000) #initialisation du Replay Memory

        self.steps_done = 0 #nombre de pas réalisé au total

    def select_action(self,state,train=True):
        sample = random.random() #nombre aléatoire entre 0 et 1
        # on diminue exponentiellement le seuil epsilon 
        # selon le nombre de pas réalisé depuis le 1er épisode :
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay) 
        self.steps_done += 1
        if train: #si en mode apprentissage
            if sample > eps_threshold:
                # VOTRE CODE
                ############
                # Calcul et renvoi de l'action fournie par le réseau
                # On calcule l'action qui donne un gain moyen plus élevé d'après le modèle :
                with torch.no_grad():
                    return self.policy_net(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
            else:
                # VOTRE CODE
                ############
                # Calcul et renvoi d'une action choisie aléatoirement
                return LongTensor([[random.randrange(2)]])
        else: #si en mode test :
            # On calcule l'action qui donne un gain moyen plus élevé d'après le modèle :
            with torch.no_grad():
                return self.policy_net(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)

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

    def optimize_model(self):

        # si la longueur de la mémoire est plus petite que la taille de batch
        # on ne fait rien :
        if len(self.memory) < self.batch_size:
            return

        # on échantillonne des observations depuis la mémoire (replay memory ):
        transitions = self.memory.sample(self.batch_size)
        # on les sépare selon ce qu'elles représentent 
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
        # et on les met au bon format :
        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward)).unsqueeze(-1)
        batch_next_state = Variable(torch.cat(batch_next_state))

        # VOTRE CODE
        ############
        # Calcul de Q(s_t,a) : Q pour l'état courant
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)

        # VOTRE CODE
        ############
        # Calcul de Q pour l'état suivant
        # on utilise le réseau cible comme expliqué dans les rappels théoriques :
        max_next_q_values = self.target_net(batch_next_state).detach().max(1)[0]
        # VOTRE CODE
        ############
        # Calcul de Q future attendue cumulée
        # équation de Bellman (rappels théoriques)
        expected_q_values = batch_reward + (self.gamma * max_next_q_values).unsqueeze(-1)

        # VOTRE CODE
        ############
        # Calcul de la fonction de perte de Huber
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        # VOTRE CODE
        ############
        # Optimisation du modèle
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_policy_model(self):

        for i_episode in range(self.num_episodes):

            state = self.env.reset()
            steps = 0

            while True:
                
                # Choix de l'action (via exploitation ou exploration)
                action = self.select_action(FloatTensor([state]))
                # observation du nouvel état et récompense obtenue
                # et si l'épisode se termine ou pas :
                next_state, reward, done, _ = self.env.step(action[0,0].item())
                
                # Définition du reward en fonction de la durée de l'épisode :
                # (on la définit différemment pour créer une "incentive" à l'agent
                # à faire durer l'épisode le plus longtemps possible)
                if done:
                    if steps < 30:
                        reward -= 10
                    else:
                        reward = -1
                if steps > 100:
                    reward += 1
                if steps > 200:
                    reward += 1
                if steps > 300:
                    reward += 1
                
                # Ajout du nouvel échantillon observé à la mémoire :
                self.memory.push((FloatTensor([state]),
                     action,
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

                # Optimisation du modèle
                self.optimize_model()

                state = next_state
                steps += 1

                # on arrête la boucle si l'épisode se termine
                # ou si une limite de 1000 est atteinte pour la durée
                # cette limite est définie pour empêcher l'épisode
                # de durer indéfiniment
                # et on la définit assez haute
                # pour qu'elle soit satisfaisante comme performance atteinte
                if done or steps >= 1000:
                    self.episode_durations.append(steps)
                    self.plot_durations()
                    break
            # Mise à jour du réseau cible à chaque période (définie au début)
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()

        # Enregistrement du modèle
        self.save_model()
        print('Training completed')
        plt.show()


    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./qlearning_model")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./qlearning_model", map_location=device))

    def test(self):
        print('Testing model:')
        for i_episode in range(self.num_episodes):
            print('episode: {}'.format(i_episode))

            state = self.env.reset()

            while True:
                self.env.render()
                # VOTRE CODE
                ############
                # Sélection d'une action appliquée à l'environnement
                # et mise à jour de l'état
                # Sélection d'une action (exploitation seulement) :
                action = self.select_action(FloatTensor([state]),train=False)
                # Mise à jour de l'état :
                next_state, _, done, _ = self.env.step(action[0,0].item())

                state = next_state

                if done:
                    break

        print('Testing completed')

if __name__ == '__main__':

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    env = gym.make('CartPole-v0').unwrapped
    env.reset()

    agent = Agent(env)

    # #Training phase
    # agent.train_policy_model()

    #Testing phase
    agent.load_model()
    agent.test()

    env.close()
