""" 
Initial code link:
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with Tensorflow

Authors: 
Patrick Emami
Shusen Wang



* The code is modified by Hongbo Fang *
"""

import tensorflow as tf
import numpy as np
from neural_network_share_weight import NeuralNetworks
from replay_buffer import ReplayBuffer
import csv
import os
import argparse
import math
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.01
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
SINGLE_TRAIN = None

RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 240
EVALUATE = False



def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                L.append(os.path.join(root, file))
    return L


class FishEnv:


    def __init__(self,reward):



        self.games = None
        self.sim_dir = './metadata/'
        self.sim_file = None
        self.valid_bots = ['1','2','3','4'] # ignore bot 0
        self.length = 480 # total number of ticks in the whole game
        self.x_size = 480.0 # the boundary length of the x axis
        self.y_size = 285.0 # the boundary length of the y axis
        self.reward = reward
        self.close_choose = ['first','second'] # Choose whether it's 'first' or 'second' in 'close' value
        self.cond_choose = ['wall','spot']  # Choose whether it's 'wall' or 'spot' in 'cond' value

        bots_positions = []

        files_sim = file_name(self.sim_dir) # get a list of all 'csv' files in that game
        for direc in files_sim:
            # read player action

            if 'simulation.csv' not in direc:
                continue
            f = open(direc)
            csv_reader = csv.reader(f)

            for row in csv_reader:
                if row[0] in self.valid_bots:
                    bots_positions.append(float(row[3])) # get the x_position of bot
                    bots_positions.append(float(row[4])) # get the y_position of bot
            f.close()
        self.bots_pos_std = np.std(np.array(bots_positions)) # calcualte the standard deviation of coordinates, used for normalization

        print("bots position std",self.bots_pos_std)



    def _normalization(self, bots, state_cur):
        '''
        normalize bots coordination and current state of agent
        Input:
            bots: state of four bots, [4, 7]
            state_cur: state of agent, [8]
        Output:
            new_bots: [4, 7], bots state with x,y coordinates and goal_x, goal_y coordinates normalized
            new_state_cur: 8, agent state with x, y coordinates and goal_x, goal_y coordinates normalized
            np.concatenate([np.reshape(new_bots,[-1]),new_state_cur]): [36], a concatenation of normalized bots state and agent state
        '''
        new_bots = np.copy(bots) 
        new_state_cur = np.copy(state_cur)
        # normalized with equation
        #     (x-mean)/ std
        # mean is replaced by the half of total length in x axis for simplicity
        # std for both x and y is set to the same for simplicity
        new_bots[:,0] -= self.x_size/2 
        new_bots[:,1] -= self.y_size/2
        new_bots[:,2] -= self.x_size/2
        new_bots[:,3] -= self.y_size/2
        new_bots[:,:4] /= self.bots_pos_std
        new_state_cur[0] -= self.x_size/2
        new_state_cur[1] -= self.y_size/2
        new_state_cur[2] -= self.x_size/2
        new_state_cur[3] -= self.y_size/2
        new_state_cur[:4] /= self.bots_pos_std

        # new_bots: 4 * 7, new_state_cur: 8
        return new_bots, new_state_cur, np.concatenate([np.reshape(new_bots,[-1]),new_state_cur])



    def reset(self,target_index = None,bg_cond = None, close_half=None):
        '''
        reset to initial stage, initialization before every game
        Input:
            target_index: reset to a given simulation number, randomly selecting if None
            bg_cond: reset to a simulation with a given bg_cond, randomly selecting if None
            close_half: reset to a simulation with a given close_half, randomly selecting if None
        '''
        self.global_time = 0 # global_time: indicate the current time step
        self.Current_Score = 0 # Current_Score: indicate the current score (not the reward, but representing whether the agent is in the scoring area or not)

        # randomly choose 'bg_cond' and 'close_half'
        self.bg_cond = self.cond_choose[np.random.randint(2)] 
        self.close_half = self.close_choose[np.random.randint(2)]


        # randomly choose 'sim_num'
        self.sim_num = np.random.randint(100)


        # set the values if sim_num, bg_cond or close_half is given
        if target_index != None:

            print("bg_cond",bg_cond)
            print("close_half",close_half)
            self.sim_num = target_index
            if bg_cond != None:
                self.bg_cond = self.cond_choose[bg_cond]
            if close_half != None:
                self.close_half = self.close_choose[close_half]


        # close_round_limit is used to define the range of in_close condition(always get score no matter what)
        if self.close_half == "first":
            self.close_round_limit = [65, 160]
        else:
            self.close_round_limit = [305, 400]
        print("reset to make the sim_num to be",self.sim_num, "bg_cond", self.bg_cond, "close_half", self.close_half)
        self.current_status = np.zeros(8)
        # 0: x_pos
        # 1: y_pos
        # 2: goal_pos_x
        # 3: goal_pos_y
        # 4,5,6 speed(4 for max speed, 5 for slow speed, 6 for zero speed, one hot for dimension 4, 5 or 6)
        # 7: reward

        # initially put agent at the center of the area
        self.current_status[0] = self.x_size / 2
        self.current_status[1] = self.y_size / 2

        # initial goal is exactly the same as the original
        self.current_status[2] = self.current_status[0]
        self.current_status[3] = self.current_status[1]

        self.current_status[6] = 1.0 # initially set to no speed
        # if it's speed 0, goal doesn't matter

        sim_prefix = self.sim_dir + 'v2-' + self.bg_cond + '-close_' + self.close_half + '-asocial-smart-0-' + str(self.sim_num)
        centers = [sim_prefix + '-social-matched_bg.csv', sim_prefix + '-social-mismatch_bg.csv']
        # center addresses


        self.sim_file = sim_prefix + '-social-simulation.csv'

        self.bots = np.zeros([4, self.length, 7])
        # cur_x_pos, cur_y_pos, norm_x_offset, norm_y_offset, speed(3)

        self.cycle = np.zeros([self.length,2,2]) # notice that players cannot observe the cycle, and we only need the x_y position
                                                 # the first 2 in the dimension is the x and y coordinates
                                                 # the last 2 in the dimension is the cycle index(0 or 1)
        f_sim = open(self.sim_file)
        csv_reader = csv.reader(f_sim)  # read simulation
        pre_xy = np.zeros([4,2])
        for row in csv_reader:

            if row[0] in self.valid_bots:
                index = int(row[0]) - 1 # bots index
                tick = int(row[1]) # current tick

                # location of bots
                self.bots[index,tick,0] = float(row[3]) 
                self.bots[index,tick,1] = float(row[4])

                if tick-1 >= 0:
                    offset_x = float(row[3]) - pre_xy[index,0]
                    offset_y = float(row[4]) - pre_xy[index,1]
                    l_t = np.linalg.norm([offset_x, offset_y]) + 1e-10
                    # set current goal
                    self.bots[index, tick, 2] = float(row[9])
                    self.bots[index, tick, 3] = float(row[10])
                    if l_t > 6:
                        # 1 for max speed
                        self.bots[index, tick-1, 4] = 1
                    elif l_t > 1.5:
                        self.bots[index, tick-1, 5] = 1
                    else:
                        self.bots[index, tick-1, 6] = 1
                else:
                    # tick == 0
                    self.bots[index, tick, 2] = self.bots[index,tick, 0]
                    self.bots[index, tick, 3] = self.bots[index,tick, 1]

                pre_xy[index,0] = self.bots[index,tick,0]
                pre_xy[index,1] = self.bots[index,tick,1]

        for circle_dir_index  in xrange(len(centers)):
            f_cycle = open(centers[circle_dir_index])
            csv_reader = csv.reader(f_cycle)  # read cycle
            row_index = 0

            for row in csv_reader:

                if row[0] !='x_pos':

                    if row[0] == "":
                        self.cycle[row_index,0,circle_dir_index] = -1.0
                        self.cycle[row_index,1,circle_dir_index] = -1.0
                    else:
                        self.cycle[row_index,0,circle_dir_index] = float(row[0])
                        self.cycle[row_index,1,circle_dir_index] = float(row[1])

                    row_index += 1
            f_cycle.close()

        bots_states, agents_state, stream_states = self._normalization(self.bots[:,self.global_time,:],self.current_status)
        debug = [0,0,0,0]
        return bots_states, agents_state, stream_states,self.current_status, debug

    def normalized_location(self,value,is_X):
        # normalize the location
        if is_X == True:
            return ( value - self.x_size/2 ) / self.bots_pos_std

        else:
            return ( value - self.y_size/2 ) / self.bots_pos_std

    def _denormalized(self,value,is_X):
        # denormalize the location
        if is_X == True:
            return value * self.bots_pos_std + self.x_size/2
        else:
            return value * self.bots_pos_std + self.y_size/2

    def step(self,action):


        # action will be of dimension 5
        # normalized goal_x
        # normalized goal_y
        # speed [3]
        self.Current_Score = 0
        self.global_time += 1
        denorm_x = self._denormalized(action[0],True)
        denorm_y = self._denormalized(action[1],False)
        offset = [denorm_x - self.current_status[0],denorm_y - self.current_status[1]]
        total_length = np.linalg.norm(offset) + 1e-6

        speed_select = np.argmax(action[2:])

        # cur_x_should is the coordinates in x axis for the next time step
        # cur_y_should is the coordiantes in y axis for the next time step
        if speed_select == 0:
            # speed 7.5
            cur_x_should = 7.5 / total_length *  offset[0] + self.current_status[0]
            cur_y_should = 7.5 / total_length *  offset[1] + self.current_status[1]
            self.current_status[2] = denorm_x
            self.current_status[3] = denorm_y
            s = 0
        elif speed_select == 1:
            # speed 2.125
            cur_x_should = 2.125 / total_length *  offset[0] + self.current_status[0]
            cur_y_should = 2.125 / total_length *  offset[1] + self.current_status[1]
            self.current_status[2] = denorm_x
            self.current_status[3] = denorm_y
            s = 1
        else:
            # speed 0
            cur_x_should = self.current_status[0]
            cur_y_should = self.current_status[1]
            self.current_status[2] = self.current_status[0]
            self.current_status[3] = self.current_status[1]
            s = 2


        # set position to be the next one
        self.current_status[0] = cur_x_should
        self.current_status[1] = cur_y_should
        # change speed
            # clear the previous speed
        self.current_status[4:7] = 0

            # set the current speed
        self.current_status[s + 4] = 1


        if self.global_time > self.close_round_limit[0] and self.global_time <= self.close_round_limit[1]:
            self.Current_Score = 1
        elif self.cycle[self.global_time,0,0] < 0:
            # There is no cycle at current time step
            self.Current_Score = 0
        else:
            # Check if the agent is within the circle(I mispell it to cycle) 
            for cycyle_num in xrange(2):
                if np.linalg.norm([self.cycle[self.global_time,0,cycyle_num] - cur_x_should, self.cycle[self.global_time,1,cycyle_num] - cur_y_should]) < 50:
                    self.Current_Score = 1
                    break

        # first set the reward to 0
        self.current_status[7] = self.reward(self.bots[:,self.global_time,:], self.current_status, speed_select, self.Current_Score, denorm_x, denorm_y)

        if self.global_time == self.length -1:
            done = True
        else:
            done = False

        # we have two objects as observation, and no Info

        # the last self.current_status works as info, and only used for test and drawing
        bots_states, agents_state, stream_states = self._normalization(self.bots[:,self.global_time,:],self.current_status)
        debug = [cur_x_should, cur_y_should, offset[0],offset[1]]
        return bots_states, agents_state, stream_states, self.current_status[7], done, self.current_status, debug


class Actor():
    
    def __init__(self, sess, network, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate
        _, self.a_dim = network.get_const()
        # self.hidden_inputs = network.get_state_in(is_target=False)
        self.inputs = network.get_input_state(is_target=False)
        self.out = network.get_actor_out(is_target=False)
        # self.hidden_out = network.get_actor_hidden_out(is_target=False)
        self.params = network.get_actor_params(is_target=False)
        self.state_feature = network.get_state_feature(is_target=False)
        # This gradient will be provided by the critic network
        self.critic_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients
        self.policy_gradient = tf.gradients(tf.multiply(self.out, -self.critic_gradient), self.params)
        
        # Optimization Op        
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.policy_gradient, self.params))
        
    def train(self, state, c_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: state,
            self.critic_gradient: c_gradient
        })

    def predict(self, state):
        return self.sess.run([self.out,self.state_feature], feed_dict={self.inputs: state})




class ActorTarget():
    
    def __init__(self, sess, network, tau):
        self.sess = sess
        self.tau = tau
        # self.hidden_inputs = network.get_state_in(is_target=True)
        self.inputs = network.get_input_state(is_target=True)
        self.out = network.get_actor_out(is_target=True)
        # self.hidden_out = network.get_actor_hidden_out(is_target=True)
        self.params = network.get_actor_params(is_target=True)
        param_num = len(self.params)
        self.params_other = network.get_actor_params(is_target=False)
        assert(param_num == len(self.params_other))
        
        # update target network
        self.update_params = \
            [self.params[i].assign(tf.multiply(self.params_other[i], self.tau) +
                                                  tf.multiply(self.params[i], 1. - self.tau))
                for i in xrange(param_num)]
    
    def train(self):
        self.sess.run(self.update_params)

    def predict(self, state):
        return self.sess.run(self.out, feed_dict={self.inputs: state})
        
        
class Critic:
    def __init__(self, sess, network, learning_rate):
        self.sess = sess
        self.learning_rate = learning_rate

        # Create the critic network
        self.state, self.action = network.get_input_state_action(is_target=False)
        self.out = network.get_critic_out(is_target=False)
        self.params = network.get_critic_params(is_target=False)
        
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        #self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.loss = tf.nn.l2_loss(self.predicted_q_value - self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)

    def train(self, state, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={
            self.state: state,
            self.action: action
        })
        
    def action_gradients(self, state, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: state,
            self.action: actions
        })
        
class CriticTarget:
    def __init__(self, sess, network, tau):
        self.sess = sess
        self.tau = tau

        # Create the critic network
        self.state, self.action = network.get_input_state_action(is_target=True)
        self.out = network.get_critic_out(is_target=True)
        
        # update target network
        self.params = network.get_critic_params(is_target=True)
        param_num = len(self.params)
        self.params_other = network.get_critic_params(is_target=False)
        assert(param_num == len(self.params_other))
        self.update_params = \
            [self.params[i].assign(tf.multiply(self.params_other[i], self.tau) + tf.multiply(self.params[i], 1. - self.tau))
                for i in xrange(param_num)]
            
    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={
            self.state: state,
            self.action: action
        })

    def train(self):
        self.sess.run(self.update_params)
        

def train(sess, network,eval, reward, TYPE):

    
    RESULTS_FILE = './results_'+TYPE+"/"

    if os.path.exists(RESULTS_FILE) == False:
        os.mkdir(RESULTS_FILE)

    env = FishEnv(reward)
    arr_reward = np.zeros(MAX_EPISODES)
    arr_qmax = np.zeros(MAX_EPISODES)

    actor = Actor(sess, network, ACTOR_LEARNING_RATE)
    actor_target = ActorTarget(sess, network, TAU)
    critic = Critic(sess, network, CRITIC_LEARNING_RATE)
    critic_target = CriticTarget(sess, network, TAU)

    s_dim, a_dim = network.get_const()

    max_q_learn = -1000.0
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=1000000)
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    if eval == True:       
        # used to generate test training set for our stage 2, no used in he 
        print("Evaluate mode")
        ckpt = tf.train.get_checkpoint_state(RESULTS_FILE)
        saver.restore(sess, ckpt.model_checkpoint_path)
        s_dim, a_dim = network.get_const()

        total_action_store = [] # this should be 40 * 480* dimension
        total_obser_store = []
        for repeat in xrange(20):
            for index in xrange(100):
                for bg_cond in xrange(2):
                    for close_half in xrange(2):
                        save_action = []
                        save_observe = []
                        bots_s, agents_s,s, info,debug = env.reset(index,bg_cond,close_half)
                        header = ['tick','x_pos','y_pos','x_should_next','y_should_next', 'goal_x', 'goal_y', 'bg_cond','close_cond','sim_num','bg_val','round_type','score']
                        csvFile = open(RESULTS_FILE+"evaluate"+str(index)+TYPE+"-b"+str(bg_cond)+"-c"+str(close_half)+"-r-"+str(repeat)+".csv", "w")
                        writer = csv.writer(csvFile)
                        writer.writerow(header)



                        done = False

                        # wirte the initial stage

                        r = 0
                        while True:

                            a,state = actor.predict(np.reshape(s, (1, s_dim))) # get the action out

                            save_observe.append(s)
                            save_action.append(a[0])
                            # calcualte the speed
                            denorm_x = env._denormalized(a[0, 0],True)
                            denorm_y = env._denormalized(a[0, 1],False)
                            total_length = np.linalg.norm([denorm_x - info[0], denorm_y - info[1]]) + 1e-6
                            speed_select = np.argmax(a[0,2:])

                            if speed_select == 0:
                                # speed 7.5
                                cur_x_should = 7.5 / total_length *  (denorm_x - info[0]) + info[0]
                                cur_y_should = 7.5 / total_length *  (denorm_y - info[1]) + info[1]
                            elif speed_select == 1:
                                cur_x_should = 2.125 / total_length *  (denorm_x - info[0]) + info[0]
                                cur_y_should = 2.125 / total_length *  (denorm_y - info[1]) + info[1]
                            else:
                                cur_x_should = info[0]
                                cur_y_should = info[1]

                            to_write = [env.global_time,info[0],info[1],cur_x_should,cur_y_should,denorm_x, denorm_y, env.bg_cond,env.close_half,env.sim_num,r,'social',env.Current_Score]
                            writer.writerow(to_write)
                            if done == True:
                                break
                            non_use1, non_use2, s, r, done, info, debug = env.step(a[0])
                        total_action_store.append(save_action)
                        total_obser_store.append(save_observe)
                        print("repeat",repeat,"type",TYPE)
                        print("shape of save_action",np.shape(np.array(save_action)))
                        print("shape of save_observe",np.shape(np.array(save_observe)))
                        csvFile.close()
        total_action_store = np.array(total_action_store)
        total_obser_store = np.array(total_obser_store)
        t_d = "./npz_train/"
        if os.path.exists(t_d) == False:
            os.mkdir(t_d)
        np.savez(t_d+TYPE+"store.npz", action = total_action_store,observe = total_obser_store)
        return
    REWARD_DIRE = "./rewards/"
    if os.path.exists(REWARD_DIRE) == False:
        os.mkdir(REWARD_DIRE)
    actor_target.train()
    critic_target.train()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    TOTAL_REWARD = []
    for i in xrange(MAX_EPISODES):
        bots_s, agents_s,s, info,debug  = env.reset() # shape [21]

        ep_reward = 0
        ep_ave_max_q = 0
        if len(TOTAL_REWARD) != 0 and len(TOTAL_REWARD)%100 == 0:
            TOTAL_REWARD = np.array(TOTAL_REWARD)
            np.save(REWARD_DIRE+"total_reward"+str(i)+".npy", TOTAL_REWARD)
            TOTAL_REWARD = []
        for j in xrange(MAX_EP_STEPS):

            # Added exploration noise

            a,_ = actor.predict(np.reshape(s, (1, s_dim)))
            a1 = np.copy(a)
            if np.random.rand() < 0.1/(i+0.1):
                # random exploration
                a[0,:] = 0
                a[0,0] = env.normalized_location(np.random.rand() * env.x_size, True) 
                a[0,1] = env.normalized_location(np.random.rand() * env.y_size, False)
                t_select = np.random.randint(3)
                a[0,2 + t_select] = 1
            non_use1, non_use2, s2, r, terminal, info, debug = env.step(a[0])
            replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                              terminal, np.reshape(s2, (s_dim,)))


            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic_target.predict(s2_batch, actor_target.predict(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                #ep_ave_max_q += np.amax(predicted_q_value)
                ep_ave_max_q += np.mean(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs,_= actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor_target.train()
                critic_target.train()

            s = s2
            ep_reward += r

            if terminal:
                TOTAL_REWARD.append(ep_reward)
                denominator = float(j) + 1e-10
                print('Reward: ' + str(ep_reward) + ',   Episode: ' + str(i) + ',    Qmax: ' +  str(ep_ave_max_q / denominator),"ep_ave_max_q",ep_ave_max_q)
                arr_reward[i] = ep_reward
                arr_qmax[i] = ep_ave_max_q / denominator
                print("previous max q_learn",max_q_learn,"current_max_q_learn",ep_ave_max_q / denominator,"type",TYPE)
                if i % 20 == 0 and ep_ave_max_q / denominator > max_q_learn:
                    max_q_learn = ep_ave_max_q / denominator
                    # np.savez(RESULTS_FILE, arr_reward[0:i], arr_qmax[0:i])
                    checkpoint_path = RESULTS_FILE
                    saver.save(sess, checkpoint_path, global_step=i)
                    print("Automatic Save Player file")
                        # we need to rewrite the csv
                    # put it in that one temporarly
                    bots_s, agents_s,s, info,debug = env.reset(68)
                    # s,info = env.reset(68)
                    header = ['tick','x_pos','y_pos','x_should_next','y_should_next','goal_x', 'goal_y', 'bg_cond','close_cond','sim_num','bg_val','round_type','score']
                    csvFile = open(RESULTS_FILE+"rl-"+'0'+'-'+str(i)+TYPE+".csv", "w")
                    Writer = csv.writer(csvFile)
                    Writer.writerow(header)


                    s_dim, a_dim = network.get_const()
                    done = False

                    # wirte the initial stage
                    r = 0
                    while True:
                        a,_ = actor.predict(np.reshape(s, (1, s_dim))) # get the action out
                        # calcualte the speed
                        denorm_x = env._denormalized(a[0, 0],True)
                        denorm_y = env._denormalized(a[0, 1],False)
                        total_length = np.linalg.norm([denorm_x - info[0], denorm_y - info[1]]) + 1e-6
                        speed_select = np.argmax(a[0,2:])

                        if speed_select == 0:
                            # speed 7.5
                            cur_x_should = 7.5 / total_length *  (denorm_x - info[0]) + info[0]
                            cur_y_should = 7.5 / total_length *  (denorm_y - info[1]) + info[1]
                        elif speed_select == 1:
                            cur_x_should = 2.125 / total_length *  (denorm_x - info[0]) + info[0]
                            cur_y_should = 2.125 / total_length *  (denorm_y - info[1]) + info[1]
                        else:
                            cur_x_should = info[0]
                            cur_y_should = info[1]

                        to_write = [env.global_time,info[0],info[1],cur_x_should,cur_y_should, denorm_x, denorm_y, env.bg_cond,env.close_half,env.sim_num,r,'social',env.Current_Score]
                        Writer.writerow(to_write)
                        if done == True:
                            break
                        # s, r, done, info = env.step(a[0])
                        non_use1, non_use2, s, r, done, info, debug = env.step(a[0])
                    csvFile.close()
                    print("Saved at batch",i)
                break
