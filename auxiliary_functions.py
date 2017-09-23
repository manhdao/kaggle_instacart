import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import itertools

from useful_lists import GESTURE_CODES_MOBILE, important_screens, bins_dfc, gesture_list_ios



def all_screens(data):
    """Additional function to iew all screens in the dataset"""

    def union_set(x,y):
        x = set(x)
        y = set(y)
        return x.union(y)
    return reduce(union_set, data['screens_visited'])




def make_screen_vector(cell, metric, list_of_screens):
    """Make a vector of interaction or time from interaction_information, based on list_of_screens"""

    vector = np.zeros(len(list_of_screens))
    for screen in cell:
        try:
            i = list_of_screens.index(screen['_screen'])
            if metric == 'interaction':
                vector[i] += screen['interaction_count']
            if metric == 'time':
                vector[i] += screen['view_time']
        except:
            continue

    return vector



def timestamp_array(cell, type, scale=False):
    """Contruct an array of timestamp of screen visits or interactions, with option to scale to [0,1]"""

    alist = []
    for screen in cell:
        if type == 'screen':
            alist.append(screen['start_time'])
        elif type == 'interaction':
            alist += list(screen['interaction_times'])
    
    an_arr = np.array(alist)

    if scale == True:
        last_screen = cell[-1]
        total_time = last_screen['start_time'] + last_screen['view_time']
        an_arr = an_arr / total_time

    
    return an_arr




def interaction_coor_selected_screen_49(obj, selected_screens=important_screens, df_mode=True, print_err=False):
    """Extract timestamp data as histogram of 1-second bins, resulting in unequal sequence length"""

    master_dict = {}
    for i in range(0,len(selected_screens)):
        master_dict[i] = []

    def partitioning_screen_49(arr, rect, bin=4, diff=False):
        """Actually count occurences of interaction in 4 partitions for array with the same label and 
        difference in x-y coors, using histogram2d"""

        w, h = rect[0], rect[1] # dimensions of physical screen

        if bin == 4:
            if len(arr) == 0:
                bin4 = np.zeros(4)
            elif len(arr) != 0:
                Bin2D = np.histogram2d(x=arr[:,1], y=arr[:,2], bins=(np.linspace(0,w,3), np.linspace(0,h,3)))[0]
                Bin2D = Bin2D.T

                if diff == True:
                    Bin2D_diff = np.histogram2d(x=arr[:,3], y=arr[:,4], bins=(np.linspace(0,w,3), np.linspace(0,h,3)))[0]
                    Bin2D += Bin2D_diff.T

                bin4 = np.array([Bin2D[0,0], Bin2D[0,1], Bin2D[1,0], Bin2D[1,1]])

            return bin4

        if bin == 9:
            if len(arr) == 0:
                bin9 = np.zeros(9)
            elif len(arr) != 0:
                Bin2D = np.histogram2d(x=arr[:,1], y=arr[:,2], bins=(np.linspace(0,w,4), np.linspace(0,h,4)))[0]
                Bin2D = Bin2D.T
                    
                if diff == True:
                    Bin2D_diff = np.histogram2d(x=arr[:,3], y=arr[:,4], bins=(np.linspace(0,w,4), np.linspace(0,h,4)))[0]
                    Bin2D += Bin2D_diff.T

                bin9 = np.array([Bin2D[0,0], Bin2D[0,1], Bin2D[0,2],
                                 Bin2D[1,0], Bin2D[1,1], Bin2D[1,2],
                                 Bin2D[2,0], Bin2D[2,1], Bin2D[2,2]])

            return bin9
        

    def binning_interaction_information_49(cell, rect, selected_screens=important_screens, print_err=print_err):
        """Extract timestamp data as histogram, equal or unequal, 
                    and extract coors data in equal bins on screen"""

        for screen in cell:
            for i in range(0, len(selected_screens)):
                if screen['_screen'] == selected_screens[i]:

                    interaction_number = screen['interaction_count']

                    # TIMESTAMP
                    interaction_time_array = np.array(screen['interaction_times']) - screen['start_time']
                    # fix recording error, where first interaction timestamp is smaller than start timestamp
                    if (interaction_time_array[0] < 0.) and (interaction_time_array[0] > -0.5): 
                        interaction_time_array[0] = 0.001

                    if screen['view_time'] > 0:
                        # fix recording error, where last interaction timestamp is greater than viewtime
                        hist4_timebins = np.linspace(0,screen['view_time'],num=5)
                        hist4_timebins[-1] += 1. 
                        hist9_timebins = np.linspace(0,screen['view_time'],num=10)
                        hist9_timebins[-1] += 1.
                        hist4 = np.histogram(interaction_time_array, bins=hist4_timebins)[0]
                                # dividing the array into 4 equal-sized bins
                        hist9 = np.histogram(interaction_time_array, bins=hist9_timebins)[0]
                                # dividing the array into 9 equal-sized bins
                    else:
                        if print_err == True:
                            print(screen['_screen'], screen['interaction_labels'], interaction_time_array, screen['view_time'])
                        continue

                    if (np.sum(hist4) != interaction_number) or (np.sum(hist9) != interaction_number):
                        if print_err == True:
                            print(screen['_screen'], screen['interaction_labels'], interaction_time_array, screen['view_time'])

                    hist_vector = [interaction_number] + hist4.tolist() + hist9.tolist() # appending the histograms together
                    hist_vector = np.array(hist_vector)


                    # LABEL
                    label_vector = np.zeros(len(gesture_list_ios)) # full vector of all labels
                    label_array = [] # converting label to 0, 1, or 2 
                    for interaction in screen['interaction_labels']:
                        j = gesture_list_ios.index(interaction)
                        label_vector[j] += 1

                        # converting label sequence to a label_array with number encoder
                        if interaction in ['single_tap', 'double_tap']:
                            label_array.append(0)
                        elif interaction in ['swipe_up','swipe_down','swipe_left','swipe_right','scroll','trail']:
                            label_array.append(1)
                        else:
                            label_array.append(2)

                    assert np.sum(label_vector) == interaction_number
                    assert len(label_array) == interaction_number
                    
                    label_3 = np.zeros(3)
                    label_3[0] = np.sum(label_vector[0:2]) # tap
                    label_3[1] = np.sum(label_vector[2:7]) + label_vector[11] # swipe
                    label_3[2] = np.sum(label_vector[7:11]) # other                      
                    
                    label_array = np.array(label_array).reshape(interaction_number,1)


                    # COORDINATES
                    
                    coors_array = screen['interaction_coors']
                    # find interactions with changes in x_y coors, aka the "trails".
                    coors1, coors2 = coors_array[:,0:2], coors_array[:,2:4]
                    diff = np.absolute(coors1 - coors2)
                    diff_pos = np.sum(diff, axis=1) > 0.
                    diff_pos = diff_pos.reshape(len(coors_array),1)

                    label_coors_array = np.concatenate((label_array, coors_array, (1*diff_pos)), axis=1) # starting array

                    bin1 = interaction_number + np.sum(1*diff_pos) # bin1 is the whole physical screen
                    
                    # separate starting array into 5 arrays of different label and difference in x-y coors
                    tap_arr = label_coors_array[label_coors_array[:,0] == 0] # only tap interaction
                    bin4_tap = partitioning_screen_49(tap_arr, rect=rect, bin=4)
                    bin9_tap = partitioning_screen_49(tap_arr, rect=rect, bin=9)

                    swipe_same_arr = label_coors_array[(label_coors_array[:,0] == 1) & (label_coors_array[:,5] == 0)] 
                                                                # only swipe interaction and no diff
                    bin4_swipe = partitioning_screen_49(swipe_same_arr, rect=rect, bin=4)
                    bin9_swipe = partitioning_screen_49(swipe_same_arr, rect=rect, bin=9)

                    swipe_diff_arr = label_coors_array[(label_coors_array[:,0] == 1) & (label_coors_array[:,5] == 1)] 
                                                                # only swipe interaction and diff
                    bin4_swipe += partitioning_screen_49(swipe_diff_arr, rect=rect, bin=4, diff = True)
                    bin9_swipe += partitioning_screen_49(swipe_diff_arr, rect=rect, bin=9, diff = True)

                    other_same_arr = label_coors_array[(label_coors_array[:,0] == 2) & (label_coors_array[:,5] == 0)] 
                                                                # only other interaction and no diff
                    bin4_other = partitioning_screen_49(other_same_arr, rect=rect, bin=4)
                    bin9_other = partitioning_screen_49(other_same_arr, rect=rect, bin=9)

                    other_diff_arr = label_coors_array[(label_coors_array[:,0] == 2) & (label_coors_array[:,5] == 1)] 
                                                                # only other interaction and diff
                    bin4_other += partitioning_screen_49(other_diff_arr, rect=rect, bin=4, diff = True)
                    bin9_other += partitioning_screen_49(other_diff_arr, rect=rect, bin=9, diff = True)

                    bin_vector = [bin1] + bin4_tap.tolist() + bin9_tap.tolist() + bin4_swipe.tolist() \
                                + bin9_swipe.tolist() + bin4_other.tolist() + bin9_other.tolist() # append all the bins together
                    assert len(bin_vector) == 40
                    bin_vector = np.array(bin_vector)


                    # APPEND ALL 3 VECTOR TIMESTAMP, LABEL, COORS TOGETHER
                    master_dict[i].append((hist_vector, label_3, bin_vector, screen['view_time']))

    if df_mode == True:
        for index, row in obj.iterrows():
            cell = row['interaction_information']
            rectangle = (row.width, row.height)
            binning_interaction_information_49(cell, rect=rectangle, selected_screens=selected_screens)

    elif df_mode == False:
        cell = obj['interaction_information']
        rectangle = (obj.width, obj.height)
        binning_interaction_information_49(cell, rect=rectangle, selected_screens=selected_screens)

    return master_dict



def interaction_time_coor_array_selected_screen(obj, selected_screens=important_screens, df_mode=True, equal_bins=True, print_err=False):
    """Extract timestamp data as histogram of 1-second bins, resulting in unequal sequence length"""

    master_dict = {}
    for i in range(0,len(selected_screens)):
        master_dict[i] = []

    def binning_interaction_information(cell, rect, selected_screens=important_screens, equal_bins=True, print_err=print_err):
        """Extract timestamp data as histogram, equal or unequal, 
                    and extract coors data in equal bins on screen"""

        for screen in cell:
            for i in range(0, len(selected_screens)):
                if screen['_screen'] == selected_screens[i]:

                    # TIMESTAMP
                    interaction_time_array = np.array(screen['interaction_times']) - screen['start_time']
                    # fix recording error, where first interaction timestamp is smaller than start timestamp
                    if (interaction_time_array[0] < 0.) and (interaction_time_array[0] > -0.5): 
                        interaction_time_array[0] = 0.001

                    if equal_bins == True:
                        hist1 = len(interaction_time_array) # bin of 1 is just the total_count

                        if screen['view_time'] > 0:
                            # fix recording error, where last interaction timestamp is greater than viewtime
                            hist4_timebins = np.linspace(0,screen['view_time'],num=5)
                            hist4_timebins[-1] += 1. 
                            hist9_timebins = np.linspace(0,screen['view_time'],num=10)
                            hist9_timebins[-1] += 1.
                            hist4 = np.histogram(interaction_time_array, bins=hist4_timebins)[0]
                                    # dividing the array into 4 equal-sized bins
                            hist9 = np.histogram(interaction_time_array, bins=hist9_timebins)[0]
                                    # dividing the array into 9 equal-sized bins
                        else:
                            if print_err == True:
                                print(screen['_screen'], screen['interaction_labels'], interaction_time_array, screen['view_time'])
                            continue

                        if (np.sum(hist4) != hist1) or (np.sum(hist9) != hist1):
                            if print_err == True:
                                print(screen['_screen'], screen['interaction_labels'], interaction_time_array, screen['view_time'])

                        hist_vector = [hist1] + hist4.tolist() + hist9.tolist() # appending the histograms together
                        hist_vector = np.array(hist_vector)

                    elif equal_bins == False: # deprecated
                        length = int(np.ceil(screen['view_time']))
                        hist_vector = np.histogram(interaction_time_array, bins=range(0,length+1))[0]

                    # LABEL
                    label_vector = np.zeros(len(gesture_list_ios))
                    for interaction in screen['interaction_labels']:
                        j = gesture_list_ios.index(interaction)
                        label_vector[j] += 1
                    assert np.sum(label_vector) == screen['interaction_count']

                    # COORDINATES
                    w, h = rect[0], rect[1] # dimensions of physical screen

                    orientation_array = screen['orientations'].reshape(len(screen['orientations']),1)
                    coors_array = screen['interaction_coors']
                    assert len(orientation_array) == len(coors_array)

                    # find interactions with changes in x_y coors, aka the "trails".
                    coors1, coors2 = coors_array[:,0:2], coors_array[:,2:4]
                    diff = np.absolute(coors1 - coors2)
                    diff_pos = np.sum(diff, axis=1) > 0.
                    diff_pos = diff_pos.reshape(len(orientation_array),1)

                    ori_coors_array = np.concatenate((orientation_array, coors_array, (1*diff_pos)), axis=1)

                    bin1 = len(orientation_array) + np.sum(1*diff_pos) # bin1 is the total physical screen

                    bin4 = np.zeros((4,)) # partition the physical screens into 4 quadrants

                    bin9 = np.zeros((9,)) # partition the physical screens into 9 quadrants

                    # find which bin the interaction belongs to, using histogram2D
                    for interaction in ori_coors_array:
                        Bin2D_4b = np.histogram2d(x=np.array([interaction[1]]), y=np.array([interaction[2]]), 
                                                        bins=(np.linspace(0,w,3), np.linspace(0,h,3)))[0]
                        Bin2D_4b = Bin2D_4b.T
                        Bin2D_9b = np.histogram2d(x=np.array([interaction[1]]), y=np.array([interaction[2]]), 
                                                        bins=(np.linspace(0,w,4), np.linspace(0,h,4)))[0]
                        Bin2D_9b = Bin2D_9b.T

                        # set up the trail bins, just in case
                        Bin2D_4b_trail = np.zeros((2,2))
                        Bin2D_9b_trail = np.zeros((3,3))


                        if interaction[5] == 1: # there is x-y coors change, aka it's a trail. Also count the tail end of the trail
                            Bin2D_4b_trail = np.histogram2d(x=np.array([interaction[3]]), y=np.array([interaction[4]]), 
                                                        bins=(np.linspace(0,w,3), np.linspace(0,h,3)))[0]
                            Bin2D_4b_trail = Bin2D_4b_trail.T
                            Bin2D_9b_trail = np.histogram2d(x=np.array([interaction[3]]), y=np.array([interaction[4]]), 
                                                        bins=(np.linspace(0,w,4), np.linspace(0,h,4)))[0]
                            Bin2D_9b_trail = Bin2D_9b_trail.T

                        # actual counting the occurences of interactions in each bin
                        if interaction[0] == 1: # portrait
                            bin4 += np.array([Bin2D_4b[0,0], Bin2D_4b[0,1], Bin2D_4b[1,0], Bin2D_4b[1,1]])
                            bin4 += np.array([Bin2D_4b_trail[0,0], Bin2D_4b_trail[0,1], Bin2D_4b_trail[1,0], Bin2D_4b_trail[1,1]])
                            bin9 += np.array([Bin2D_9b[0,0], Bin2D_9b[0,1], Bin2D_9b[0,2],
                                              Bin2D_9b[1,0], Bin2D_9b[1,1], Bin2D_9b[1,2],
                                              Bin2D_9b[2,0], Bin2D_9b[2,1], Bin2D_9b[2,2]])
                            bin9 += np.array([Bin2D_9b_trail[0,0], Bin2D_9b_trail[0,1], Bin2D_9b_trail[0,2],
                                              Bin2D_9b_trail[1,0], Bin2D_9b_trail[1,1], Bin2D_9b_trail[1,2],
                                              Bin2D_9b_trail[2,0], Bin2D_9b_trail[2,1], Bin2D_9b_trail[2,2]])

                        if interaction[0] == 0: # landscape right
                            bin4 += np.array([Bin2D_4b[0,1], Bin2D_4b[1,1], Bin2D_4b[0,0], Bin2D_4b[1,0]])
                            bin4 += np.array([Bin2D_4b_trail[0,1], Bin2D_4b_trail[1,1], Bin2D_4b_trail[0,0], Bin2D_4b_trail[1,0]])
                            bin9 += np.array([Bin2D_9b[0,2], Bin2D_9b[1,2], Bin2D_9b[2,2],
                                              Bin2D_9b[0,1], Bin2D_9b[1,1], Bin2D_9b[2,1],
                                              Bin2D_9b[0,0], Bin2D_9b[1,0], Bin2D_9b[2,0]])
                            bin9 += np.array([Bin2D_9b_trail[0,2], Bin2D_9b_trail[1,2], Bin2D_9b_trail[2,2],
                                              Bin2D_9b_trail[0,1], Bin2D_9b_trail[1,1], Bin2D_9b_trail[2,1],
                                              Bin2D_9b_trail[0,0], Bin2D_9b_trail[1,0], Bin2D_9b_trail[2,0]])


                        if interaction[0] == 2: # landscape left
                            bin4 += np.array([Bin2D_4b[1,0], Bin2D_4b[0,0], Bin2D_4b[1,1], Bin2D_4b[0,1]])
                            bin4 += np.array([Bin2D_4b_trail[1,0], Bin2D_4b_trail[0,0], Bin2D_4b_trail[1,1], Bin2D_4b_trail[0,1]])
                            bin9 += np.array([Bin2D_9b[2,0], Bin2D_9b[1,0], Bin2D_9b[0,0],
                                              Bin2D_9b[2,1], Bin2D_9b[1,1], Bin2D_9b[0,1],
                                              Bin2D_9b[2,2], Bin2D_9b[1,2], Bin2D_9b[0,1]])
                            bin9 += np.array([Bin2D_9b_trail[2,0], Bin2D_9b_trail[1,0], Bin2D_9b_trail[0,0],
                                              Bin2D_9b_trail[2,1], Bin2D_9b_trail[1,1], Bin2D_9b_trail[0,1],
                                              Bin2D_9b_trail[2,2], Bin2D_9b_trail[1,2], Bin2D_9b_trail[0,1]])

                        if interaction[0] == 4: # portrait upside-down
                            bin4 += np.array([Bin2D_4b[1,1], Bin2D_4b[1,0], Bin2D_4b[0,1], Bin2D_4b[0,0]])
                            bin4 += np.array([Bin2D_4b_trail[1,1], Bin2D_4b_trail[1,0], Bin2D_4b_trail[0,1], Bin2D_4b_trail[0,0]])
                            bin9 += np.array([Bin2D_9b[2,2], Bin2D_9b[2,1], Bin2D_9b[2,0],
                                              Bin2D_9b[1,2], Bin2D_9b[1,1], Bin2D_9b[1,0],
                                              Bin2D_9b[0,2], Bin2D_9b[0,1], Bin2D_9b[0,0]])
                            bin9 += np.array([Bin2D_9b_trail[2,2], Bin2D_9b_trail[2,1], Bin2D_9b_trail[2,0],
                                              Bin2D_9b_trail[1,2], Bin2D_9b_trail[1,1], Bin2D_9b_trail[1,0],
                                              Bin2D_9b_trail[0,2], Bin2D_9b_trail[0,1], Bin2D_9b_trail[0,0]])

                    bin_vector = [bin1] + bin4.tolist() + bin9.tolist() # appending the bins together
                    bin_vector = np.array(bin_vector)

                    master_dict[i].append((hist_vector, label_vector, bin_vector, screen['view_time']))


    if df_mode == True:
        for index, row in obj.iterrows():
            cell = row['interaction_information']
            rectangle = (row.width, row.height)
            binning_interaction_information(cell, rect=rectangle, selected_screens=selected_screens, equal_bins=equal_bins)

    elif df_mode == False:
        cell = obj['interaction_information']
        rectangle = (obj.width, obj.height)
        binning_interaction_information(cell, rect=rectangle, selected_screens=selected_screens, equal_bins=equal_bins)

    return master_dict



def extract_interaction_sequence_selected_screen(df, selected_screens=important_screens, equal_bins=True, 
                                                    print_err=False, coor_details=True):
    """Build a new dataframe with interaction histogram sequence and screen""" 

    df_seq_list = []
        
    for index, row in df.iterrows():
        if coor_details == False:
            row_dict = interaction_time_coor_array_selected_screen(row, selected_screens=selected_screens, 
                                                                df_mode=False, equal_bins=equal_bins, print_err=print_err)
        elif coor_details == True:
            row_dict = interaction_coor_selected_screen_49(row, selected_screens=selected_screens, 
                                                                df_mode=False, print_err=print_err)

        for key, value in row_dict.items():
            if len(value) > 0:
                for array_quad in value:
                    df_seq_dict = {'session_id': row['id'],
                                   'user_id': row['device_id'],
                                   'time':row['time_formatted'],
                                   'screen': selected_screens[key],
                                   'interaction_hist': array_quad[0],
                                   'interaction_label': array_quad[1],
                                   'interaction_coors_binned': array_quad[2],
                                   'view_time': array_quad[3]
                                  }
                    df_seq_list.append(df_seq_dict)
    
    df_seq = pd.DataFrame(df_seq_list,columns=['session_id','user_id','time','screen','interaction_hist','interaction_label',
                                               'interaction_coors_binned','view_time'])

    return df_seq



def build_overall_vector(row):
    """Concat all vectors of df_seq together into a super vector"""

    hist_arr = row.interaction_hist.tolist()
    label_arr = row.interaction_label.tolist()
    bin_arr = row.interaction_coors_binned.tolist()
    viewtime = [row.view_time]

    return [hist_arr + label_arr + bin_arr + viewtime]
    



def count_screen_in_sessions_data(df, list_of_screens, top=5):
    """Calculate the percentage of appearance in screens in sessions data"""

    count_matrix = np.zeros((len(df),len(list_of_screens)))
    i=0
    for index, row in df.iterrows():
        cell = row['interaction_information']
        row_count = make_screen_vector(cell, 'interaction', list_of_screens=list_of_screens)
        row_count_pos = row_count > 0
        row_count_ones = row_count_pos * 1
        count_matrix[i] = row_count_ones
        i += 1
    
    count_sum = count_matrix.sum(axis=0)
    
    count_sort = np.sort(count_sum)[::-1]

    for j in range(top):
        i = count_sum.tolist().index(count_sort[j])
        print(list_of_screens[i] +' appears in %s of sessions' %('{:.1%}'.format(count_sum[i]/len(df))))



def distribution_cluster_result(label_cluster):
    """Print distritbution of clusering results"""
    
    n_clusters_ = len(set(label_cluster)) - (1 if -1 in label_cluster else 0)
    print('Number of clusters:', n_clusters_)
    unique, counts = np.unique(label_cluster, return_counts=True)
    print('Distribution:' + '\n',np.asarray((unique, counts)).T)




####### DEPRECATED #######



def orientation_selected_screen(obj, selected_screens=important_screens, df_mode=True, equal_bins=True, print_err=False):
    """Check orientation of screen period interactions"""

    master_dict = {}
    for i in range(0,len(selected_screens)):
        master_dict[i] = []

    def orientation_information(cell, selected_screens=important_screens):
        """Extract orientation data"""

        for screen in cell:
            for i in range(0, len(selected_screens)):
                if screen['_screen'] == selected_screens[i]:

                    orientation_array = screen['orientations']
                    master_dict[i].append(orientation_array)


    if df_mode == True:
        for index, row in obj.iterrows():
            cell = row['interaction_information']
            rectangle = (row.width, row.height)
            orientation_information(cell, selected_screens=selected_screens)

    elif df_mode == False:
        cell = obj['interaction_information']
        rectangle = (obj.width, obj.height)
        orientation_information(cell, selected_screens=selected_screens)

    return master_dict




def extract_orientation_selected_screen(df, selected_screens=important_screens):
    """Build a new dataframe with orientation_data""" 

    df_seq_list = []
        
    for index, row in df.iterrows():
        row_dict = orientation_selected_screen(row, selected_screens=selected_screens, df_mode=False)

        for key, value in row_dict.items():
            if len(value) > 0:
                for array in value:
                    df_seq_dict = {'session_id': row['id'],
                                   'user_id': row['device_id'],
                                   'time':row['time_formatted'],
                                   'screen': selected_screens[key],
                                   'orientation_list': array,
                                  }
                    df_seq_list.append(df_seq_dict)
    
    df_seq = pd.DataFrame(df_seq_list,columns=['session_id','user_id','time','screen','orientation_list'])

    return df_seq



def interaction_dataframe_selected_screen(df, screens=important_screens, bins=bins_dfc): # deprecated
    """Create a df from interaction histogram over time, using selected screens, 
    sharing same bins to have equal number of features"""

    def interaction_array_selected_screen(df, screens=screens, bins=bins):
        """Extract array information from interaction_information column"""

        master_dict = {}
        for i in range(0,len(screens)):
            master_dict[i] = []

        for cell in df['interaction_information']:
            for screen in cell:
                for i in range(0, len(screens)):
                    if screen['_screen'] == screens[i]:
                        interaction_time_array = np.array(screen['interaction_times']) - screen['start_time']
                        hist = np.histogram(interaction_time_array, bins=bins)[0]
                        master_dict[i].append(hist)

        return master_dict
    
    master_dict = interaction_array_selected_screen(df)

    # Parse everything into dataframe
    X = pd.DataFrame()
    y = []

    for i in range(0,len(master_dict)):
        X = X.append(pd.DataFrame(master_dict[i]),ignore_index=True)
        y += len(master_dict[i]) * [i]

    return X.to_sparse(fill_value=0), np.array(y)


def in_list(cell, list_of_items=[]): # deprecated
    """Count the number of times an item in list_of_items appear"""

    sum_list = 0
    for ele in cell:
        if ele in list_of_items:
            sum_list += 1
    return sum_list



def scale_threshold(series, threshold, max_score = 1.): # deprecated
    """Min-max scale a series into a scale of 10, with values > threshold becoming 10"""
    
    series_threshold = series.copy()
    series_threshold[series_threshold > threshold] = threshold
    series_scaled = max_score * (series_threshold - min(series_threshold)) / (threshold - min(series_threshold))
    return series_scaled



def find_latest_interaction(cell):
    """Extract the last timestamp of interaction on each screen, and find the max in the session"""

    last_timestamps = []
    for screen in cell:
        last_interaction_time = screen['interaction_times'][-1] - screen['start_time']
        last_timestamps.append(last_interaction_time)
    return max(last_timestamps)



def explicit_interactions(cell): # deprecated
    """Extract gestures & interactions from 'cor' in each cell, excluding UI screens"""
    
    def remove_wrong_swipe(list1):
        list2=[]
        if len(list1) >= 2:
            for i in range(0, len(list1)-1):
                if (not list1[i].startswith('swipe')) or (list1[i+1] != 'trail'):
                    list2.append(list1[i])
            list2.append(list1[-1])
        else:
            list2 = list1.copy()
        return list2

    def extract_interactions(ele):
        adict = {}
        adict['_screen'] = ele['an']
        adict['all_interactions'] = [GESTURE_CODES_MOBILE[x[3]] for x in ele['cor']]
        adict['all_true_interactions'] = remove_wrong_swipe(adict['all_interactions'])
        adict['interaction_count'] = len(adict['true_interactions'])
        adict['start_time'] = ele['at']
        adict['view_time'] = ele['vt']
        return adict

    alist = []
    for ele in cell:
            adict = extract_interactions(ele)
            alist.append(adict)
    
    return alist