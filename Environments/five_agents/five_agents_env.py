import pstats
import random, math, os
from xml.dom.expatbuilder import parseString
import numpy as np
from enum import Enum

import sys
sys.path.append('../')
sys.path.append('../../')
from reward_machines.sparse_reward_machine import SparseRewardMachine

"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none 


class FiveAgentEnv:

    def __init__(self, rm_file, agent_id, env_settings, my_shared_events, individual_events):
        """
        Initialize environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        agent_id : int
            Index {0,1,...} indicating which agent is being trained in this environment.
        env_settings : dict
            Dictionary of environment settings
        """
        #print(rm_file)
        self.env_settings = env_settings
        self.agent_id = agent_id 
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file) # I do want this
        #print(self.reward_machine)
        self.agent_events_dict = {0: {'ei': 'e1'}, 1: {'ei': 'e2'}, 2: {'ei': 'e3'}, 3: {'ei': 'e4'}, 4: {'ei': 'e5'}}
    
        self.u = self.reward_machine.get_initial_state()
        self.last_action = -1 # Initialize last action to garbage value
        self.my_shared_events  = my_shared_events
        self.individual_events = individual_events

        self.flags = {'e1': False, 'e2': False, 'e3': False, 'e4': False, 'e5': False, 'a': False, 'b': False, 'c': False , 'done': False, 'failed': False}

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        initial_states = self.env_settings['initial_states']

        self.s_i = initial_states[self.agent_id]

        self.objects = {}
        for e, e_loc in self.env_settings['button_locs']:
            self.objects[e_loc] = e # goal location
    
        self.p = self.env_settings['p']

        self.num_states = self.Nr * self.Nc

        self.actions = [Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.none.value]
        
        # Define forbidden transitions corresponding to map edges
        self.forbidden_transitions = set()
        
        wall_locations = self.env_settings['walls']

        for row in range(self.Nr):
            self.forbidden_transitions.add((row, 0, Actions.left)) # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, self.Nc - 1, Actions.right)) # If in right-most column, can't move right.
        for col in range(self.Nc):
            self.forbidden_transitions.add((0, col, Actions.up)) # If in top row, can't move up
            self.forbidden_transitions.add((self.Nr - 1, col, Actions.down)) # If in bottom row, can't move down

        # Restrict agent from having the option of moving "into" a wall
        for i in range(len(wall_locations)):
            (row, col) = wall_locations[i]
            self.forbidden_transitions.add((row, col + 1, Actions.left))
            self.forbidden_transitions.add((row, col-1, Actions.right))
            self.forbidden_transitions.add((row+1, col, Actions.up))
            self.forbidden_transitions.add((row-1, col, Actions.down))

    def environment_step(self, s, a):
        """
        Execute action a from state s.

        Parameters
        ----------
        s : int
            Index representing the current environment state.
        a : int
            Index representing the action being taken.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : list
            List of events occuring at this step.
        s_next : int
            Index of next state.
        """
        s_next, last_action = self.get_next_state(s,a) 
        self.last_action = last_action 

        row, col = self.get_state_description(s_next)

        l = self.get_mdp_label_update_fix(s, s_next, self.u) # I think I have this correct... (yikes)
        r = 0
        ei = self.agent_events_dict[self.agent_id]['ei']

        end_thresh = self.env_settings['end_thresh']

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e) #check what this does for undefined events. TODO: 
            if u2 in self.reward_machine.T:
                if self.random_success(end_thresh):
                     self.flags['done'] = True
                     #print("might randomly end episode")
                     #print("You thought you were done so I let you wander for a bit then ended it. ")
            r_step = self.reward_machine.get_reward(self.u, u2) # I should be quitting here if reward is negative.
            if r_step < 0: 
                self.flags['failed'] = True
                self.flags['done'] = True
            r = r +  r_step

            # Update the reward machine state
            self.u = u2 # okay cool I get this code 
            #print(f" e is {e} and tri is {tri}")
            if e == ei:
                #print("WHY AM I NOT HERE")
                self.flags[ei] = True
            if e == 'a':
                self.flags['a'] = True
            if e == 'b':
                self.flags['b'] = True
            if e == 'c':
                self.flags['c'] = True

        if l == []:
            if self.u in self.reward_machine.T:
                # probably never get here
                #print("might randomly end episode")
                if self.random_success(end_thresh):
                     self.flags['done'] = True
            if len(self.reward_machine.T) == 0:
                if self.random_success(end_thresh):
                     self.flags['done'] = True
        

        return r, l, s_next


   
    
    ##################   Labeling   ######################  
    
    def random_success(self, thresh):
        if np.random.random() <= thresh:
            return True
        return False

   
    def get_mdp_label_update_fix(self, s, s_next, u):
        if u in self.reward_machine.delta_u.keys():
            possible_transitions = self.reward_machine.delta_u[u].keys()
        else:
            possible_transitions = []
        #print(f"The transitions I can take are: {possible_transitions} .")

        ei = self.agent_events_dict[self.agent_id]['ei']

        thresh = self.env_settings['thresh'] 
        l = []
        row, col = self.get_state_description(s_next)

        # What order do I check:
        # li
        if ei in possible_transitions:
            if (row, col) == self.env_settings['button_locs'][ei]:
                l.append(ei)
        if 'a' in possible_transitions:
            if self.flags['a'] == False:
                if 'e1' in self.individual_events:
                    if self.flags['e1'] == True:
                        if 'a' in self.my_shared_events:
                            if self.random_success(thresh):
                                l.append('a') #randomly simulate? 
                        else: 
                            l.append('a')
                else:
                    if 'a' in self.my_shared_events:
                        if self.random_success(thresh):
                            l.append('a') #randomly simulate? 
        if 'b' in possible_transitions:
            if self.flags['b'] == False:
                if 'e2' in self.individual_events:
                    if self.flags['e2'] == True:
                        if 'b' in self.my_shared_events:
                            if self.random_success(thresh):
                                l.append('b') #randomly simulate? 
                        else: 
                            l.append('b')     
                else:
                    if 'b' in self.my_shared_events:
                            if self.random_success(thresh):
                                l.append('b') #randomly simulate? 
                    else: 
                        l.append('b')
        return l[:1]
     
    ######################################################

    def get_next_state(self, s, a):
        """
        Get the next state in the environment given action a is taken from state s.
        Update the last action that was truly taken due to MDP slip.

        Parameters
        ----------
        s : int
            Index of the current state.
        a : int
            Action to be taken from state s.

        Outputs
        -------
        s_next : int
            Index of the next state.
        last_action :int
            Last action taken by agent due to slip proability.
        """
        slip_p = [self.p, (1-self.p)/2, (1-self.p)/2]
        check = random.random()

        row, col = self.get_state_description(s)

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

        if (check<=slip_p[0]) or (a == Actions.none.value):
            # No slip for picking up and putting down ? 
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0: 
                a_ = 3
            elif a == 2: 
                a_ = 1
            elif a == 3: 
                a_ = 2
            elif a == 1: 
                a_ = 0

        else:
            if a == 0: 
                a_ = 1
            elif a == 2: 
                a_ = 3
            elif a == 3: 
                a_ = 0
            elif a == 1: 
                a_ = 2

        action_ = Actions(a_)

        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                row -= 1
            if action_ == Actions.down:
                row += 1
            if action_ == Actions.left:
                col -= 1
            if action_ == Actions.right:
                col += 1 

        s_next = self.get_state_from_description(row, col)

        last_action = a_
        return s_next, last_action

    def get_state_from_description(self, row, col):
        """
        Given a (row, column) index description of gridworld location, return
        index of corresponding state.

        Parameters
        ----------
        row : int
            Index corresponding to the row location of the state in the gridworld.
        col : int
            Index corresponding to the column location of the state in the gridworld.
        
        Outputs
        -------
        s : int
            The index of the gridworld state corresponding to location (row, col).
        """
        return self.Nc * row + col

    def get_state_description(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """
        row = np.floor_divide(s, self.Nr)
        col = np.mod(s, self.Nc)

        return (row, col)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.last_action

    def get_initial_state(self):
        """
        Outputs
        -------
        s_i : int
            Index of agent's initial state.
        """
        return self.s_i

    def is_complete(self):
        if self.flags['done'] == True:
            return True
        elif self.flags['failed'] == True:
            return True
        return False 

    def reset(self):
        """ 
        Start environment over to natural state 
        """
        self.flags = {'tree': True, 'a1': False, 'a2': False, 'a3': False, 'log': False, 'carrying': False, 'arrived': False, 'crafted': False, 'failed': False }

    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc), dtype=int)
        
        # Display the locations of the walls
        for loc in self.env_settings['walls']:
            display[loc] = -1

    
        for ei, ei_loc in self.env_settings['button_locs'].items():
            display[ei_loc] = 10* int(ei[-1])

        # Display the location of the agent in the world
        row, col = self.get_state_description(s)
        display[row,col] = self.agent_id + 1

        print(display)


def play(my_shared_events, individual_events, agent_id, file_name = 'crafting_rm_1.txt'):

    base_file_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    #rm_string = os.path.join(base_file_dir,'automated_task_assignment_with_rm', 'data', 'saved_reward_machines', 'crafting_task', 'crafting_env_testing', file_name) #need to load my game 
    #print("RM STRING", rm_string)
    rm_string = os.path.join( 'data', 'saved_reward_machines', 'five_agent', 'five_agent_testing', file_name) #need to load my game 

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 15
    env_settings['Nc'] = 15
    env_settings['initial_states'] = [0, 10, 210, 224, 111] 
    env_settings['walls'] = [(0,2), (1,2), (2,2), (2,3), (2,4), (5,1), (5,2), (5,3), (6,3), (7,3), (8,3), (9,3), (9,2), (9,1), (9,0), (11,0), (0,6), (1,6), (2,6), (4,6), (5,6), (5,5), (5,7), (5,8), (5,9), (5,10), 
     (6,7), (7,7), (8,7), (10,7), (11,7), (11,6), (11,8), (12,8), (13,8), (14,8), (0,9), (1,9), (2,9), (2,10), (2,11), (2,12), (2,13), (4,13), (5,13), (6,13), (8,11), (8,12), (8,13), (8,14), (12,11), (12, 12), (12,13), (12,14)]
    
    button_locs = {'e1': (0,3), 'e2': (10,14), 'e3': (13,6), 'e4': (1,12), 'e5': (7,0)}
    env_settings['button_locs'] = button_locs
    env_settings['thresh'] = .3
    env_settings['end_thresh'] = .3
    env_settings['p'] = 0.99

    game = FiveAgentEnv(rm_string, agent_id, env_settings, my_shared_events, individual_events)
    # User inputs
    
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value,"x":Actions.none.value} # add pick up and drop? 
    s = game.get_initial_state()
    full_l = []
    
    while True:
        # Showing game
        game.show(s)
        print("Individual Events", game.individual_events)
        print("Shared Events for Me", game.my_shared_events)
        # Getting action
        print("\nAction? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action:
            r, l, s = game.environment_step(s, str_to_action[a])
            full_l = full_l + l
            
            print("---------------------")
            print("Next States: ", s)
            print("Label: ", l)
            print("Reward: ", r)
            print("RM state: ", game.u)
            print("task flags: ", game.flags)
            print('full trace:', full_l)
            print("---------------------")

            if game.flags['failed']:
                break
            if game.flags['done']: # Game Over # need to change my game over criteria (Change in learning?)
                break 
        else:
            print("Forbidden action")

    game.show(s)
    


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()