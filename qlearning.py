import tensorflow as tf
import gym
import numpy as np

class LinearPolicy(object):
    """Implements Q function as linear function of state"""
    def __init__(self, state_shape, action_shape):
        self.W = np.random.randn(action_shape, state_shape+1)
        self.target = np.random.randn(action_shape, state_shape+1)


    def choose(self, state):
        t = self.W.dot(np.concatenate((np.array([1]),state),axis = 0))
        return np.argmax(t), t

    def choose_target(self, state):
        t = self.target.dot(np.concatenate((np.array([1]),state),axis = 0))
        return np.argmax(t), t



    def train(self,env, lr = 0.001, discount = 0.9, n_evals_per_model = 10):
        # evaluate this episode
        done = False
        c = n_evals_per_model
        reward = 0
        states = []
        s = env.reset()
        while c > 0:
            a, q = self.choose(s)
            sp, run, done, _ = env.step(a)
            y = run + np.max(self.choose_target(sp)[1]) - q
            loss = np.linalg.norm(y - q)**2
            x_state = np.concatenate((np.array([1]), s), axis = 0)
            dW = lr*np.outer(y-q,x_state) # step_size
            self.W += dW
            env.render()

            if done:
                s = env.reset()
                self.target = self.W
                c -= 1

        reward /= n_evals_per_model
        


env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    state_shape=4
    action_shape = env.action_space.n

    agent = LinearPolicy(state_shape, action_shape)
    agent.train(env)
