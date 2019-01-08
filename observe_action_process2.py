
import csv
import numpy as np
valid_bots = ["1","2","3","4"]
SEQUENCE_L = 480
def player_state_action_process(player_file_direc,sim_dir):
    # player_dires = "./player-data/"

    x_size = 480
    y_size = 285
    # action 5
    # normalized goal_x
    # normalized goal_y
    # speed 2 ( note we don't have speed = 0, so we consider speed = 0 to be speed = 2.125)

    # observe 32
    # 4 * 7
    # cur_x_pos, cur_y_pos, norm_x_offset, norm_y_offset, speed(3)
    # cur_x_pos and cur_y pos is normalized

    # current status 4
    # 0: x_pos(normalized)
    # 1: y_pos(normalized)
    # 2: normalized goal_x
    # 3: normalized goal_y
    # 4-6: previous speeds
    # 7: reward
    # 8: wall

    # assumes it has been opened

    csv_reader = csv.reader(open(player_file_direc))
    player_content = []
    for row in csv_reader:
        if row[0] != "pid":
            player_content.append(row)
    bg_cond = player_content[0][-3]
    close_half = player_content[0][-2]
    sim_num = player_content[0][-1]

    sim_prefix = sim_dir + 'v2-' + bg_cond + '-close_' + close_half + '-asocial-smart-0-' + sim_num
    bots_file_direc = sim_prefix + '-social-simulation.csv'
    # read bots file


    bots_pos_std = 128.7 # it's fixed number

    csv_reader = csv.reader(open(bots_file_direc)) # reopen ?
    pre_xy = np.zeros([4,2])

    bots = np.zeros([SEQUENCE_L, 4 , 7],dtype=np.float32)
    actions = np.zeros([SEQUENCE_L,5],dtype=np.float32)



    for row in csv_reader:

        if row[0] in valid_bots:
            index = int(row[0]) - 1
            tick = int(row[1])

            bots[tick, index,0] = float(row[3])
            bots[tick, index,1] = float(row[4])

            if tick-1 >= 0:
                offset_x = float(row[3]) - pre_xy[index,0]
                offset_y = float(row[4]) - pre_xy[index,1]
                l_t = np.linalg.norm([offset_x,offset_y]) + 1e-10

                bots[tick, index, 2] = offset_x / l_t
                bots[tick, index, 3] = offset_y / l_t
                if l_t > 6:
                    # 1 for max speed
                    bots[tick, index, 4] = 1
                elif l_t > 1.5:
                    bots[tick, index, 5] = 1
                else:
                    bots[tick, index, 6] = 1
            pre_xy[index,0] = bots[tick, index,0]
            pre_xy[index,1] = bots[tick, index,1]



    cur_status = np.zeros([SEQUENCE_L,8])
    for tick in xrange(SEQUENCE_L):

        next_p_xy = np.array([float(player_content[tick+1][3]), float(player_content[tick+1][4])]) if tick<SEQUENCE_L-1 else []
        last_p_xy = np.array([float(player_content[tick-1][3]), float(player_content[tick-1][4])]) if tick >= 1 else []
        cur_p_xy = np.array([float(player_content[tick][3]), float(player_content[tick][4])])

        speed_next = np.linalg.norm(next_p_xy - cur_p_xy) if len(next_p_xy) >0 else 0
        speed_last = np.linalg.norm(last_p_xy - cur_p_xy) if len(last_p_xy) >0 else 0


        actions[tick,0] = float(player_content[tick][10])
        actions[tick,1] = float(player_content[tick][11])

        if speed_next > 6:
            actions[tick,2] = 1
        elif speed_next > 1.5:
            actions[tick,3] = 1
        else:
            actions[tick,4] = 1

        if speed_last > 6:
            cur_status[tick,4] = 1
        elif speed_last > 1.5:
            cur_status[tick,5] = 1
        else:
            cur_status[tick,6] = 1

        cur_status[tick,0] = float(player_content[tick][3]) - x_size / 2.0
        cur_status[tick,1] = float(player_content[tick][4]) - y_size / 2.0

        if  cur_status[tick,6] != 1:
            # we need to set the angle
            cur_status[tick,2] = float(player_content[tick][10]) - x_size / 2.0
            cur_status[tick,3] = float(player_content[tick][11]) - y_size / 2.0
        else:
            cur_status[tick,2] = float(player_content[tick][3]) - x_size / 2.0
            cur_status[tick,3] = float(player_content[tick][4]) - y_size / 2.0

        cur_status[tick,7] = float(player_content[tick][7])
        # cur_status[tick,8] = float(player_content[tick][12])

    bots[:,:,0] -= x_size / 2.0
    bots[:,:,1] -= y_size / 2.0
    actions[:,0] -= x_size / 2.0
    actions[:,1] -= y_size / 2.0
    for tick in xrange(SEQUENCE_L):

        bots[tick,:,:2] /= bots_pos_std
        actions[tick,:2] /= bots_pos_std
        cur_status[tick,:4] /= bots_pos_std

    res = np.concatenate([np.reshape(bots,[SEQUENCE_L,-1]),cur_status],1)

    print("shape of res",np.shape(res))
    print("shape of actions",np.shape(actions))
    player_id_first_4 = int(player_content[0][0][:4])
    print("player_id_first_4",player_id_first_4)
    return res, actions,player_id_first_4



