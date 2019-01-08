
import os
import numpy as np
import csv
def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                L.append(os.path.join(root, file))
    return L

class FishEnv:


    def __init__(self):



        self.games = None
        self.sim_dir = './metadata/'
        self.sim_file = None
        self.valid_bots = ['1','2','3','4']
        self.length = 480
        self.x_size = 480
        self.y_size = 285

        self.close_choose = ['first','second']
        self.cond_choose = ['wall','spot']

        bots_positions = []

        files_sim = file_name(self.sim_dir)
        for direc in files_sim:
            # read player action

            if 'simulation.csv' not in direc:
                continue
            f = open(direc)
            csv_reader = csv.reader(f)

            for row in csv_reader:
                if row[0] in self.valid_bots:
                    bots_positions.append(float(row[3]))
                    bots_positions.append(float(row[4]))
            f.close()
        self.bots_pos_std = np.std(np.array(bots_positions))

        print("bots position std",self.bots_pos_std)



    def _normalization(self, bots, state_cur):

        new_bots = bots.copy()
        new_state_cur = state_cur.copy()
        new_bots[:,0] -= self.x_size/2
        new_bots[:,1] -= self.y_size/2
        new_bots[:,:2] /= self.bots_pos_std
        new_state_cur[0] -= self.x_size/2
        new_state_cur[1] -= self.y_size/2
        new_state_cur[2] -= self.x_size/2
        new_state_cur[3] -= self.y_size/2
        new_state_cur[:4] /= self.bots_pos_std

        # new_bots: 4 * 7, new_state_cur: 9
        return new_bots, new_state_cur, np.concatenate([np.reshape(new_bots,[-1]),new_state_cur])



    def reset(self,target_index = None,bg_cond = None, close_half=None):
        # reset to initial stage
        self.global_time = 0


        self.bg_cond = self.cond_choose[np.random.randint(2)]
        self.close_half = self.close_choose[np.random.randint(2)]


            # keep running until it's not 68
        self.sim_num = np.random.randint(100)

        if target_index != None:
            print("bg_cond",bg_cond)
            print("close_half",close_half)
            self.sim_num = target_index
            self.bg_cond = 'wall' if bg_cond == None else self.cond_choose[bg_cond]
            self.close_half = 'first' if close_half == None else self.close_choose[close_half]

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
        # 4,5,6 speed
        # 7: reward

        # randomly put agents at different places
        self.current_status[0] = self.x_size / 2
        self.current_status[1] = self.y_size / 2

        self.current_status[2] = self.current_status[0]
        self.current_status[3] = self.current_status[1]

        self.current_status[6] = 1.0 # initially set to no speed


        sim_prefix = self.sim_dir + 'v2-' + self.bg_cond + '-close_' + self.close_half + '-asocial-smart-0-' + str(self.sim_num)
        centers = [sim_prefix + '-social-matched_bg.csv', sim_prefix + '-social-mismatch_bg.csv']
        # center addresses


        self.sim_file = sim_prefix + '-social-simulation.csv'

        # num_inputs: input channel, should be 3, one for himself, one for others, one for reward or not
        # input image size: 3 * 488 * 292

        self.bots = np.zeros([4, self.length, 7])
        # cur_x_pos, cur_y_pos, norm_x_offset, norm_y_offset, speed(3)

        self.cycle = np.zeros([self.length,2,2]) # notice that players cannot observe the cycle, and we only need the x_y position
                                                 # the last 2 is the cycle index(0 or 1)
        f_sim = open(self.sim_file)
        csv_reader = csv.reader(f_sim)  # read simulation
        pre_xy = np.zeros([4,2])
        for row in csv_reader:

            if row[0] in self.valid_bots:
                index = int(row[0]) - 1
                tick = int(row[1])

                self.bots[index,tick,0] = float(row[3])
                self.bots[index,tick,1] = float(row[4])

                if tick-1 >= 0:
                    offset_x = float(row[3]) - pre_xy[index,0]
                    offset_y = float(row[4]) - pre_xy[index,1]
                    l_t = np.sqrt(np.square(offset_x) + np.square(offset_y) + 1e-10)
                    self.bots[index, tick, 2] = offset_x / l_t
                    self.bots[index, tick, 3] = offset_y / l_t
                    if l_t > 6:
                        # 1 for max speed
                        self.bots[index, tick, 4] = 1
                    elif l_t > 1.5:
                        self.bots[index, tick, 5] = 1
                    else:
                        self.bots[index, tick, 6] = 1
                pre_xy[index,0] = self.bots[index,tick,0]
                pre_xy[index,1] = self.bots[index,tick,1]

        for circle_dir_index  in range(len(centers)):
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
        if is_X == True:
            return ( value - self.x_size/2 ) / self.bots_pos_std

        else:
            return ( value - self.y_size/2 ) / self.bots_pos_std

    def _denormalized(self,value,is_X):
        if is_X == True:
            return value * self.bots_pos_std + self.x_size/2
        else:
            return value * self.bots_pos_std + self.y_size/2

    def step(self,action):


        # action will be 5
        # normalized goal_x
        # normalized goal_y

        # speed will be 3 dimension
        self.global_time += 1
        denorm_x = self._denormalized(action[0],True)
        denorm_y = self._denormalized(action[1],False)
        offset = [denorm_x - self.current_status[0],denorm_y - self.current_status[1]]
        total_length = np.linalg.norm(offset) + 1e-6

        speed_select = np.argmax(action[2:])

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


        if cur_x_should <= 0 or cur_x_should >= self.x_size or cur_y_should <= 0 or cur_y_should >= self.y_size:
            # touch the wall
            # change speed to 0
            # keep status the same

            # change reward to -1 if hitting a wall
            self.current_status[7] = -1 # set the reward to 0
            # self.current_status[8] = 1 # hitting a wall, set the flag


        else:

            # set position to be the next one
            self.current_status[0] = cur_x_should
            self.current_status[1] = cur_y_should
            # change speed
                # clear the previous speed
            self.current_status[4] = 0
            self.current_status[5] = 0
            self.current_status[6] = 0
                # set the current speed
            self.current_status[s + 4] = 1

            # first set the reward to 0
            self.current_status[7] = 0
            # self.current_status[8] = 0
            if self.cycle[self.global_time,0,0] < 0:
                # there is no cycle showing
                self.current_status[7] = 0
            else:

                for cycyle_num in range(2):
                    # 2 cycles in total
                    if np.linalg.norm([self.cycle[self.global_time,0,cycyle_num] - cur_x_should, self.cycle[self.global_time,1,cycyle_num] - cur_y_should]) < 50:
                        # we get a hit
                        self.current_status[7] = 1
                        break

            # if it doesn't touch the wall and the close_round is on ,set the reward to be 1
            if self.global_time > self.close_round_limit[0] and self.global_time <= self.close_round_limit[1]:
                self.current_status[7] = 1

            if speed_select == 2 and self.current_status[7] == 0:
                # if speed = 0 and choose to stop, deduct 0.1 points
                self.current_status[7] = -0.1

        if self.global_time == self.length -1:
            done = True
        else:
            done = False

        # we have two objects as observation, and no Info

        # the last self.current_status works as info, and only used for test and drawing
        bots_states, agents_state, stream_states = self._normalization(self.bots[:,self.global_time,:],self.current_status)
        debug = [cur_x_should, cur_y_should, offset[0],offset[1]]
        return bots_states, agents_state, stream_states, self.current_status[7], done, self.current_status, debug
