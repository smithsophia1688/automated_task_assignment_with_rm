import os
class SparseRewardMachine:
    def __init__(self,file=None):
        # <U,u0,delta_u,delta_r>
        self.U = []       # list of machine states
        self.events = set() # set of events
        self.u0 = None    # initial state
        self.delta_u = {} # state-transition function
        self.delta_r = {} # reward-transition function
        self.T = set()    # set of terminal states (they are automatically detected)
        self.equivalence_class_name_dict = {} # dict {name: equivalence class} Used in projections only 

        if file is not None:
            #print('loading file')
            #cwd = os.getcwd()  # Get the current working directory (cwd)
            #files = os.listdir(cwd)  # Get all the files in that directory
            #print("Files in %r: %s" % (cwd, files))
            self._load_reward_machine(file)
        
    def __repr__(self):
        s = "MACHINE:\n"
        s += "init: {}\n".format(self.u0)
        for trans_init_state in self.delta_u:
            for event in self.delta_u[trans_init_state]:
                trans_end_state = self.delta_u[trans_init_state][event]
                s += '({} ---({},{})--->{})\n'.format(trans_init_state,
                                                        event,
                                                        self.delta_r[trans_init_state][trans_end_state],
                                                        trans_end_state)
        if self.equivalence_class_name_dict != {}:
            s += 'State Names for Equivalence Classes: \n'
            s += str(self.equivalence_class_name_dict) + '\n'
        return s

    # Public methods -----------------------------------

    def load_rm_from_file(self, file):
        self._load_reward_machine(file)

    def get_initial_state(self):
        return self.u0

    def get_next_state(self, u1, event):
        if u1 in self.delta_u:
            if event in self.delta_u[u1]:
                return self.delta_u[u1][event]
        return u1

    def get_reward(self,u1,u2,s1=None,a=None,s2=None):
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            return self.delta_r[u1][u2]
        return 0 # This case occurs when the agent falls from the reward machine

    def get_rewards_and_next_states(self, s1, a, s2, event):
        rewards = []
        next_states = []
        for u1 in self.U:
            u2 = self.get_next_state(u1, event)
            rewards.append(self.get_reward(u1,u2,s1,a,s2))
            next_states.append(u2)
        return rewards, next_states

    def get_states(self):
        return self.U

    def is_terminal_state(self, u1):
        return u1 in self.T

    def get_events(self):
        return self.events

    def is_event_available(self, u, event):
        is_event_available = False
        if u in self.delta_u:
            if event in self.delta_u[u]:
                is_event_available = True
        return is_event_available

    def get_origins(self, u_set):
        all_origins = set()

        for state, transitions in self.delta_u.items():
            for u in u_set:
                if u in transitions.values():
                    all_origins.add(state)

        return all_origins 


    # NEW METHOD
    def write_rm_file(self, file_location, file_name):
        '''
        file_name : string with destination of file 
        '''
        file = file_location + file_name

        #print("FILE IS ", file)
        
        #print('Get current working directory : ', os.getcwd())
        with open(file, 'w+') as f:
            s = str(self.u0) + '  # Initial state \n'
            f.write(s)
            for trans_init_state in self.delta_u:
                for event in self.delta_u[trans_init_state]:
                    trans_end_state = self.delta_u[trans_init_state][event]
                    r = self.delta_r[trans_init_state][trans_end_state]

                    rm_line = f"({trans_init_state}, {trans_end_state}, '{event}', {r}) \n"
                    f.write(rm_line)
    



    # Private methods -----------------------------------

    def _load_reward_machine(self, file):
        """
        Example:
            0                  # initial state
            (0,0,'r1',0)
            (0,1,'r2',0)
            (0,2,'r',0)
            (1,1,'g1',0)
            (1,2,'g2',1)
            (2,2,'True',0)

            Format: (current state, next state, event, reward)
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        # adding transitions
        for e in lines[1:]:
            self._add_transition(*eval(e))
            self.events.add(eval(e)[2]) # By convention, the event is in the spot #indexed by 2
        # adding terminal states
        for u1 in self.U:
            if self._is_terminal(u1):
                self.T.add(u1)
        self.U = sorted(self.U)

    def calculate_reward(self, trace):
        total_reward = 0
        current_state = self.get_initial_state()

        for event in trace:
            next_state = self.get_next_state(current_state, event)
            reward = self.get_reward(current_state, next_state)
            total_reward += reward
            current_state = next_state
        return total_reward

    def _is_terminal(self, u1):
        # Check if reward is given for reaching the state in question
        for u0 in self.delta_r:
            if u1 in self.delta_r[u0]:
                if self.delta_r[u0][u1] != 0: # Small change: had been == 1. 
                    return True
        return False
            
    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U:
                self.U.append(u)

    def _add_transition(self, u1, u2, event, reward):
        # Adding machine state
        self._add_state([u1,u2])
        # Adding state-transition to delta_u
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        if event not in self.delta_u[u1]:
            self.delta_u[u1][event] = u2
        else:
            raise Exception('Trying to make rm transition function non-deterministic.')
            # self.delta_u[u1][u2].append(event)
        # Adding reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        self.delta_r[u1][u2] = reward
    
    def add_transition_and_reward_only(self, u1, u2, event, reward):
        ''' 
        Similar to _add_transition, this function is public and called when you project and parallelize reward machines. 
        Differences between these two functions:
            add_transition expects:
                (u1, u2) to already be in the state space
                event to already be in the event space

        Inputs:
            u1: element of rm.U
            u2: element of rm.U
            event: string, element of rm.events
            reward: int or float (probably 0 or 1)
        '''
        # add transition 
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        if event in self.delta_u[u1]:
            if self.delta_u[u1][event] != u2:
                raise Exception('Trying to make rm transition function non-deterministic.')
        if event not in self.delta_u[u1]:
            self.delta_u[u1][event] = u2

        # Adding reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        self.delta_r[u1][u2] = reward
    
    def get_name_from_class(self, cl):
        '''
        This function finds the "name" of a state that corresponds to a 
        particular class. 

        k is a key in self.equivalence_class_name_dict 
        if (k, cl) is an item of self.equivalence_class_name_dict, return k
        Inputs: 
            cl: (set), element of an equivalence class, value in self.equivalence_class_name_dict
       
        '''
        dict = self.equivalence_class_name_dict
        name_full = [k for k in dict if dict[k]==cl] # should have len = 1

        if len(name_full) == 0:
            sr = 'class ' + str(cl) + 'was not in the equivalence_class_name_dict for this reward machine'
            raise Exception(sr)

        return name_full[0]

