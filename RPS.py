# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np

def player(prev_play, opponent_history=[]):
    if (prev_play!=""):
        opponent_history.append(prev_play)
    # opponent_history.append(prev_play)
    options = ["R", "P", "S"]
    correct_plays = {"R":"P", "P":"S", "S":"R"}
    next_opp_play = None
    if (len(opponent_history) > 1):
        if (len(opponent_history) == 2):
            next_opp_play = calc_probs(opponent_history, 1)
        else:
            if (len(opponent_history) < 1000):
                if (len(opponent_history)<3):
                    next_opp_play = "R"
                else:
                    next_opp_play = calc_probs(opponent_history, 4)
                    
            elif (len(opponent_history) > 999 and len(opponent_history) < 1999):
                next_opp_play = calc_probs(opponent_history, 2)
                # if (len(opponent_history) < 1002):
                #     #next_opp_play = calc_probs(opponent_history, 1)
                #     next_opp_play = opponent_history[-1]
                # else:
                #     # print(opponent_history)
                #     next_opp_play = calc_probs(opponent_history[999:], 4)
                #     #next_opp_play = correct_plays[opponent_history[-3]]
                #     #return next_opp_play
            elif (len(opponent_history) > 1998 and len(opponent_history) < 2998):
                next_opp_play = calc_probs(opponent_history[1995:], 2)
            elif (len(opponent_history) > 2997):
                next_opp_play = calc_probs(opponent_history[2997:], 1)
        # next_opp_play = np.argmax(probs)
        if (next_opp_play is not None):
            our_play = correct_plays[next_opp_play]
            return our_play
    our_current_move = np.random.choice(options, 1, p=[0.3, 0.3, 0.4])[0]
    return our_current_move

def calc_probs(history: list, window_size: int = 1):
    #print(len(history))
    try:
        main_options = ["R", "P", "S"]
        R_next = {}
        P_next = {}
        S_next = {}
        epsilon = 1e-10
        second_order = ["RR", "RP", "RS", "PP", "PR", "PS", "SS", "SR", "SP"]
        third_order = []
        fourth_order = []
        for i in main_options:
            for secondOrderItem in second_order:
                third_order.append(f"{i}{secondOrderItem}")
        for i in main_options:
            for thirdOrderItem in third_order:
                fourth_order.append(f"{i}{thirdOrderItem}")
        if window_size == 1:
            for item in ["R", "P", "S"]:
                R_next[item] = 0
                P_next[item] = 0
                S_next[item] = 0
        elif window_size == 2:
            for item in second_order:
                R_next[item] = 0
                P_next[item] = 0
                S_next[item] = 0
        elif window_size == 3:
            for item in third_order:
                R_next[item] = 0
                P_next[item] = 0
                S_next[item] = 0
        elif window_size == 4:
            for item in fourth_order:
                R_next[item] = 0
                P_next[item] = 0
                S_next[item] = 0
        
        for i in range(window_size, len(history)):
            past = history[i-window_size:i]
            past_val = "".join(past)
            if(history[i] == "R"):
                R_next[past_val] += 1
            elif (history[i] == "P"):
                P_next[past_val] += 1
            else:
                S_next[past_val] += 1
        last_play = "".join(history[-window_size:])
        if (history[-1] == ""):
            last_play = "".join(history[-window_size-1:])
        if (window_size == 2):
            options = second_order
        elif (window_size == 3):
            options = third_order
        elif (window_size == 4):
            options = fourth_order
        else:
            options = main_options
        options_map = {i:j for j,i in enumerate(options)}
        total_obs_r = sum(R_next.values()) + epsilon

        #probs_R = [R_next[last_play]/total_obs_r, R_next[last_play]/total_obs_r, R_next[last_play]/total_obs_r]
        probs_R = [R_next[i]/total_obs_r for i in options]

        total_obs_p = sum(P_next.values()) + epsilon
        probs_P = [P_next[i]/total_obs_p for i in options]

        #probs_P = [P_next[last_play]/total_obs_p, P_next[last_play]/total_obs_p, P_next[last_play]/total_obs_p]
        total_obs_s = sum(S_next.values()) + epsilon
        probs_S = [S_next[i]/total_obs_s for i in options]
        #option_idx = options_map[last_play]
        #probs_S = [S_next[last_play]/total_obs_s, S_next[last_play]/total_obs_s, S_next[last_play]/total_obs_s]
        #prob_lst = [probs_R[option_idx], probs_P[option_idx], probs_S[option_idx]]
        if len(history)%100 == 0:
            # print(prob_lst)
            #if (len(history) > 1000 and len(history)<2001):
                #print(R_next)
                #print(P_next)
                #print(S_next)
            # print(history)
            # print(last_play)
            # print(window_size)
            # print(option_idx)
            # print(options_map)
            pass
        prob_lst = [R_next[last_play], P_next[last_play], S_next[last_play]]
        next_opp_play = np.argmax(prob_lst)
        return main_options[next_opp_play]
    except:
        print(history)

class MarkovChain:
    def __init__(self, unique_states : np.ndarray) -> None:
        self.unique_states = unique_states
        self.total_states = self.unique_states.shape[1]
        self.transition_probs = np.zeros(shape=(self.total_states, self.total_states))
        self.initial_probs = np.zeros_like(self.unique_states, dtype=np.float32)
        pass

    def build_transition_matrix_from_sequence(self, sequence: np.ndarray) -> None:
        for i in range(1, sequence.shape[1]):
            current_state = sequence[0, i]
            prev_state = sequence[0, i-1]
            self.transition_probs[prev_state, current_state] += 1
        # print(self.transition_probs)
        for i in range(self.total_states):
            norm_factor = np.sum(self.transition_probs[i, :]) + 1e-10
            # print(norm_factor)
            self.transition_probs[i, :] /= norm_factor
        # print(self.transition_probs)
        for i in range(self.total_states):
            state = self.unique_states[0,i]
            total_state_count = np.where(sequence == state, True, False).sum()
            self.initial_probs[0, i] = total_state_count/sequence.shape[1]
        #self.initial_probs[0,0] = 0.33 
        #self.initial_probs[0,1] = 0.33 
        #self.initial_probs[0,2] = 0.34 
        #print(self.initial_probs)
    
    def calculate_t_step_prob(self, start_state: int, target_state: int, t: int = 1) -> np.float32:
        if (t == 1):
            return np.dot(self.initial_probs, self.transition_probs[start_state, target_state])
        else:
            x = self.transition_probs
            for _ in range(1, t+1):
                x = np.dot(x, x)
            return x[start_state, target_state]

    def markov_chain_walk_prediction(self, start_state: int, walk_length: 1) -> int:
        final_state_probs = np.zeros_like(self.initial_probs)
        for i in range(self.total_states):
            final_state_probs[0, i] = self.calculate_t_step_prob(start_state=start_state,
                                                                 target_state=self.total_states[i],
                                                                 t=walk_length)
        return np.argmax(final_state_probs, axis=1)
    
    def calculate_sequence_prob(self, sequence: np.ndarray):
        prob = self.initial_probs[0, sequence[0, 0]]
        # print(prob)
        for i in range(1, sequence.shape[1]):
            prob *= self.transition_probs[sequence[0, i-1], sequence[0, i]]
        return prob

def with_mm(history: list, look_back: int, take_second: bool = False):
    options = ["R", "P", "S"]
    state_map = {"R":0, "P":1, "S":2}
    states = np.asarray([0, 1, 2], dtype=np.int32).reshape((1, 3))
    sequence = [state_map[i] for i in history]
    sequence_nd = np.asarray(sequence, dtype=np.int32).reshape((1, len(history)))
    mm = MarkovChain(states)
    mm.build_transition_matrix_from_sequence(sequence=sequence_nd)
    sequence_0 = np.concatenate((sequence_nd[:, -look_back:], np.asarray([0]).reshape((1,1))), axis=1) 
    prob_R = mm.calculate_sequence_prob(sequence_0)

    sequence_1 = np.concatenate((sequence_nd[:, -look_back:], np.asarray([1]).reshape((1,1))), axis=1) 
    prob_P = mm.calculate_sequence_prob(sequence_1)

    sequence_2 = np.concatenate((sequence_nd[:, -look_back:], np.asarray([2]).reshape((1,1))), axis=1) 
    prob_S = mm.calculate_sequence_prob(sequence_2)
    prob_arr = np.asarray([prob_R, prob_P, prob_S]).reshape((1, 3))
    most_prob = np.argmax(prob_arr, axis=1)
    second_most_prob = np.argsort(prob_arr, axis=1)
    # if ((second_most_prob[0][-1] - second_most_prob[0][-2]) < 1e-5):
    #     return options[second_most_prob[0][np.random.choice([-1, -2], 1, p=[0.5, 0.5])[0]]]
    #print(second_most_prob)
    if (len(history)<10):
        # print(history)
        # print(sequence_nd)
        # print(sequence_0)
        # print(sequence_1)
        # print(sequence_2)
        # print(mm.transition_probs)
        # print(prob_arr)
        # print(options[second_most_prob[0][-1]])
        pass

    if (len(history)%100 == 0):
        #print(sequence)
        #print(len(history))
        # print(history[-10:])
        # print(prob_arr)
        #print(most_prob)
        # print(second_most_prob)
        # print(options[second_most_prob[0][-2]])
        # print(mm.initial_probs)
        pass
    #correct_plays = {"R":"P", "P":"S", "S":"R"}
    #return options[most_prob[0]]
    if take_second:
        return options[second_most_prob[0][-2]]
    return options[second_most_prob[0][-1]]
    
