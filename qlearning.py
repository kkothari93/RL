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



    def train(self,env, lr = 0.0001, discount = 0.9, n_evals_per_model = 10):
        # evaluate this episode
        done = False
        reward = 0
        c = n_evals_per_model
        s = env.reset()
        for c in range(n_evals_per_model):
            a, q = self.choose(s)
            sp, run, done, _ = env.step(a)
            y = run + discount*np.max(self.choose_target(sp)[1]) - q
            loss = np.linalg.norm(y - q)**2
            x_state = np.concatenate((np.array([1]), s), axis = 0)
            dW = lr*np.outer(y-q,x_state) # step_size
            self.W += dW
            reward += run 
            env.render()

            if done:
                s = env.reset()
                self.target = self.W
                print("Loss, reward, episode %d = %f, %f"%(c,
                                                       reward, loss))
                reward = 0

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    state_shape=4
    action_shape = env.action_space.n

    agent = LinearPolicy(state_shape, action_shape)
    agent.train(env)
