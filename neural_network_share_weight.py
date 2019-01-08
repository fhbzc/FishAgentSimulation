""" 
Define neural network structures of the actor and critic method

The actor and critic networks share the layers: State ==> FC ==> ReLU ==> Feature

The algorithm is tested on the Pendulum-v0 OpenAI gym task 

Author: Shusen Wang
"""
import tensorflow as tf
import numpy as np


class NeuralNetworks:
    '''
    State_Feature: State to Feature
    Action_Feature: Action to Feature
    Actor: State_Feature to Action
    Critic: (State_Feature, Action_Feature) to Value
    '''
    
    def __init__(self, state_dim, action_dim):
        # size of layers
        self._S_FEATURE_DIM = 64
        self._ACTOR_H1_DIM = 128
        self._CRITIC_H1_DIM = 128
        # self._LSTM_HIDDEN = lstm_dim
        # constants
        self._S_DIM = state_dim
        self._A_DIM = action_dim

        # Create actor network
        # features extracted from states
        self.input_state, self.state_feature = self._create_state_feature(sigma=0.3)

        # self.actor_state = tf.placeholder(dtype=tf.float32, shape=[None,2 * self._LSTM_HIDDEN])
        # self.actor_state_target = tf.placeholder(dtype=tf.float32, shape=[None,2 * self._LSTM_HIDDEN])

        param_state_feature = tf.trainable_variables()
        # actor network
        self.actor_y = self._create_actors(self.state_feature)
        # parameters of actor network
        self.actor_params = tf.trainable_variables()
        num_params1 = len(tf.trainable_variables())
        
        # Create actor target network
        # features extracted from states
        self.input_state_target, self.state_feature_target = self._create_state_feature(sigma=0.3)
        param_state_feature_target = tf.trainable_variables()[num_params1:]
        # actor target network
        self.actor_target_y = self._create_actors(self.state_feature_target)
        # parameters of actor target network
        self.actor_target_params = tf.trainable_variables()[num_params1:]
        num_params2 = len(tf.trainable_variables())
        
        # Create critic network
        # features extracted from states and actions
        self.input_state_critic = self.input_state
        self.state_feature_critic = self.state_feature
        self.input_action = tf.placeholder(tf.float32, [None, self._A_DIM])
        # critic network
        self.critic_y = self._create_critics(self.state_feature_critic, self.input_action)
        # parameters of critic network
        self.critic_params = param_state_feature + tf.trainable_variables()[num_params2:]
        num_params3 = len(tf.trainable_variables())
        
        
        # Create critic target network
        # features extracted from states and actions
        self.input_state_critic_target = self.input_state_target
        self.state_feature_critic_target = self.state_feature_target
        self.input_action_target  = tf.placeholder(tf.float32, [None, self._A_DIM])
        # critic target network
        self.critic_target_y = self._create_critics(self.state_feature_critic_target, self.input_action_target)
        # parameters of critic target network
        self.critic_target_params = param_state_feature_target + tf.trainable_variables()[num_params3:]
        
    def get_const(self):
        return self._S_DIM, self._A_DIM

    def get_state_feature(self,is_target = False):
        if is_target == True:
            return self.state_feature_target
        else:
            return self.state_feature

    def get_input_state(self, is_target=False):
        if is_target:
            return self.input_state_target
        else:
            return self.input_state

    def get_actor_out(self, is_target=False):
        if is_target:
            return self.actor_target_y
        else:
            return self.actor_y
        
    def get_actor_params(self, is_target=False):
        if is_target:
            return self.actor_target_params
        else:
            return self.actor_params
        
    
    def get_input_state_action(self, is_target=False):
        if is_target:
            return (self.input_state_critic_target, self.input_action_target)
        else:
            return (self.input_state_critic, self.input_action)
    
    def get_critic_out(self, is_target=False):
        if is_target:
            return self.critic_target_y
        else:
            return self.critic_y
        
    def get_critic_params(self, is_target=False):
        if is_target:
            return self.critic_target_params
        else:
            return self.critic_params
        
    # =========== Define Networks Structures =========== #
    def _create_state_feature(self, sigma=0.1):
        '''
        State ==> FC ==> ReLU ==> Feature
        '''
        x = tf.placeholder(tf.float32, [None, self._S_DIM])
        # w = self.weight_variable([37, 64])
        w1 = tf.Variable(tf.truncated_normal([self._S_DIM, self._S_FEATURE_DIM], stddev=sigma))
        # w = self.weight_variable([self._S_DIM, self._S_FEATURE_DIM], value = sigma, rand='normal')
        # b = self.bias_variable([self._S_FEATURE_DIM])
        b1 = tf.Variable(tf.constant(0.0, shape = [self._S_FEATURE_DIM]))
        w2 = tf.Variable(tf.truncated_normal([self._S_FEATURE_DIM, self._S_FEATURE_DIM], stddev=sigma))
        b2 = tf.Variable(tf.constant(0.0, shape = [self._S_FEATURE_DIM]))
        # y = tf.nn.relu(tf.matmul(x, w) + b)
        y1 = tf.nn.tanh(tf.matmul(x, w1) + b1) 
        y = tf.nn.tanh(tf.matmul(y1, w2) + b2)
        return x, y
    
    def _create_actors(self, feature):
        '''
        State_Feature ==> FC ==> ReLU ==> FC ==> Tanh ==> Scale ==> Action
        '''
        w1 = tf.Variable(tf.truncated_normal([self._S_FEATURE_DIM, self._ACTOR_H1_DIM], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.01, shape=[self._ACTOR_H1_DIM]))
        # h1 = tf.nn.relu(tf.matmul(feature, w1) + b1)
        h1 = tf.nn.tanh(tf.matmul(feature, w1) + b1)

        w2 = tf.Variable(tf.truncated_normal([self._ACTOR_H1_DIM, self._A_DIM], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.01, shape=[self._A_DIM]))
        h2 = tf.matmul(h1, w2) + b2
        y0 = tf.nn.tanh(h2[:,0:1]) * 240.0/128.71
        y1 = tf.nn.tanh(h2[:,1:2]) * 142.5/128.71

        # y1 = tf.nn.tanh(h2[:,:2])
        y2 = tf.nn.softmax(h2[:,2:])

        return tf.concat([y0, y1 ,y2],1)
    
        
    def _create_critics(self, s_feature, action):
        '''
        Hidden Layer: 
            H1 = State_Feature * W1s + Action * W1a + Bias
        Critic:
            H1 ==> ReLU ==> FC ==> Value
        '''
        # w1_s = self.weight_variable([self._S_FEATURE_DIM, self._CRITIC_H1_DIM], value=0.3, rand='normal')
        w1_s = tf.Variable(tf.truncated_normal([self._S_FEATURE_DIM, self._CRITIC_H1_DIM], stddev=0.3))

        # w1_a = self.weight_variable([self._A_DIM, self._CRITIC_H1_DIM])
        w1_a = tf.Variable(tf.truncated_normal([self._A_DIM, self._CRITIC_H1_DIM], stddev=0.1))
        # b1 = self.bias_variable([self._CRITIC_H1_DIM], value=0.0)
        b1 = tf.Variable(tf.constant(0.0, shape = [self._CRITIC_H1_DIM]))
        h1 = tf.add(tf.matmul(s_feature, w1_s), tf.matmul(action, w1_a))
        # y1 = tf.nn.relu(tf.add(h1, b1))
        y1 = tf.nn.tanh(tf.add(h1, b1))

        # w2 = self.weight_variable([self._CRITIC_H1_DIM, 1], value=0.01, rand='uniform')
        w2 = tf.Variable(tf.random_uniform([self._CRITIC_H1_DIM, 1], minval = -0.01, maxval = 0.01))
        b2 = tf.Variable(tf.constant(0.0, shape = [1]))

        # b2 = self.bias_variable([1], value=0.0)
        y2 = tf.matmul(y1, w2) + b2
        return y2
