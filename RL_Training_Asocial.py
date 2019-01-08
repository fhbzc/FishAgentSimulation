
import tensorflow as tf
import numpy as np
import os
import argparse
from train_RL import *


def Reward(bots, current,speed_select,current_score,denorm_x,denorm_y):
    '''
    input list:
    bots: 4*7, statuses of other bots
    current: 8, status of agent at previous time step
    speed_select: the speed of current action, 0 for max speed, 1 for minimal speed, 2 for zero speed
    current_score: whether in scoring area or not
    denorm_x: non-normalized goal x
    denorm_y: non-normalized goal y
    '''
    if current_score == 1:
        if speed_select == 2:
            return 1
        elif speed_select == 0:
            return -0.3
    elif speed_select == 2:
        return -1
    elif speed_select == 1:
        return -0.1
    return 0



parser = argparse.ArgumentParser()
parser.add_argument("-e", help="whether to evaluate or not, default is None(no evaluation)",default = None,dest= 'evaluate')
parser.add_argument("-t", help="targted index",default = None,dest='target')
args = parser.parse_args()

if args.evaluate != None:
    # evaluate
    EVALUATE = True
    if args.target == None:
        # if target not set, set to 0
        SINGLE_TRAIN = 0
    else:

        SINGLE_TRAIN = int(args.target)
else:
    EVALUATE = False
    # train
    if args.target != None:
        SINGLE_TRAIN = int(args.t)
    else:
        SINGLE_TRAIN = None
def main(_):



    with tf.Session() as sess:
        print("Note: current version doesn't take in_close round into consideration")
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)


        state_dim = 36 # 4 * 7 + 9
        action_dim = 5
        # Ensure action bound is symmetric


        network = NeuralNetworks(state_dim, action_dim)
        


        train(sess, network, EVALUATE, Reward, "asocial")


if __name__ == '__main__':
    tf.app.run()
