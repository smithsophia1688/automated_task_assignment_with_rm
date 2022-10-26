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


class Line_Of_Buttons_Env:

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
        self.num_buttons = env_settings['num_buttons']
        self.agent_id = agent_id 
        self.chop_count = 0
        self.chop_thresh = 3
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file) # I do want this
        #print(self.reward_machine)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = -1 # Initialize last action to garbage value
        self.my_shared_events  = my_shared_events
        self.individual_events = individual_events

        self.flags= {'b1': False, 'b2': False, 'b3': False, 'b4': False, 'b5': False, 'b6': False, 'b7': False, 'b8': False, 'b9': False, 'b10': False, 'failed' : False, 'complete': False}
        self.buttons = ['b'+ str(i+1) for i in range(self.num_buttons)]
        self.agent_was_at_button = {'b1': False, 'b2': False, 'b3': False, 'b4': False, 'b5': False, 'b6': False, 'b7': False, 'b8': False, 'b9': False, 'b10': False, 'failed' : False} # does not start on any buttons. 

        

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        initial_states = self.env_settings['initial_states']

        self.s_i = initial_states[self.agent_id]

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

        l = self.get_mdp_label_harder_2(s, s_next, self.u) 

        r = 0
        end_thresh = self.env_settings['end_thresh']
        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e) #check what this does for undefined events. TODO: 
            if u2 in self.reward_machine.T:
                print("You think you are done but we will wait for a bit if r >=0")
                if self.random_success(end_thresh):
                     self.flags['complete'] = True
                     print("You thought you were done so I let you wander for a bit then ended it. ")
            
            r_step = self.reward_machine.get_reward(self.u, u2) # I should be quitting here if reward is negative.
            if r_step < 0: 
                self.flags['failed'] = True
            r = r +  r_step
            # Update the reward machine state
            self.u = u2 
        if l == []:
            if self.u in self.reward_machine.T:
                print("You think you are done but we will wait for a bit if r >=0")
                if self.random_success(end_thresh):
                     self.flags['complete'] = True
                     print("You thought you were done so I let you wander for a bit then ended it. ")
            
        return r, l, s_next


    ##############   Labeling Toolkit   ##################
    def random_success(self, thresh):
        if np.random.random() <= thresh:
            return True
        return False

    def button_ready(self, bi):
        '''
        This function should check if all the previous buttons have been pressed
        '''
        for b in self.buttons:
            if (b != bi) and (self.flags[b] == False):
                return False
            if (b == bi) and (self.flags[b] == False):
                return True

    def check_agent_was_at_button(self, bi):
        return self.agent_was_at_button[bi]

    def get_mdp_label(self,  s, s_next, u):
        '''
        problems with this:
        no "leaving" so no real way to explain why an agent has to wait but don't worry about it. 
        '''

        print('a')
        possible_transitions = self.reward_machine.delta_u[u].keys()
        print(f"The transitions I can take are: {possible_transitions} .")
        l = []

        row, col = self.get_state_description(s_next)
        
        for b in self.buttons:
            abi = str(self.agent_id + 1) + b
            lbi = str(self.agent_id + 1) + 'l' + b[1:]
            bi_on = b + '_on'
            if self.button_ready(b):
                if abi in possible_transitions:
                    if (row, col) == self.env_settings['button_locs'][b]:
                        l.append(abi)
                        #self.agent_was_at_button[b] = True
                elif lbi in possible_transitions:
                    if (row, col) != self.env_settings['button_locs'][b]:
                        if self.check_agent_was_at_button(b):
                            l.append(lbi)
                            #self.agent_was_at_button[b] = False
                if bi_on in possible_transitions:
                    if self.check_agent_was_at_button(b):
                        if (row, col) == self.env_settings['button_locs'][b]:
                            l.append(bi_on)
                    elif bi_on in self.my_shared_events:
                        if self.random_success(self.env_settings['thresh']):
                            l.append(bi_on)

        real_l = l[:1]
        if len(real_l) ==1: 
            for b in self.buttons: 
                abi = str(self.agent_id + 1) + b
                lbi = str(self.agent_id + 1) + 'l' + b[1:]
                bi_on = b + '_on'
                if real_l[0] == abi:
                    self.agent_was_at_button[b] = True
                elif real_l[0] == lbi:
                    self.agent_was_at_button[b] = False
                elif real_l[0] == bi_on:
                    self.flags[b] = True    

        return real_l
     
    def get_mdp_label_harder(self,  s, s_next, u):
        '''
        problems with this:
        no "leaving" so no real way to explain why an agent has to wait but don't worry about it. 
        '''
        possible_transitions = self.reward_machine.delta_u[u].keys()
        print(f"The transitions I can take are: {possible_transitions} .")
        l = []

        row, col = self.get_state_description(s_next)
        
        for b in self.buttons:
            abi = str(self.agent_id + 1) + b
            lbi = str(self.agent_id + 1) + 'l' + b[1:]
            bi_on = b + '_on'

            if abi in possible_transitions:
                if (row, col) == self.env_settings['button_locs'][b]:
                    l.append(abi)
                    #self.agent_was_at_button[b] = True
            elif lbi in possible_transitions:
                if (row, col) != self.env_settings['button_locs'][b]:
                    if self.check_agent_was_at_button(b):
                        l.append(lbi)
                        #self.agent_was_at_button[b] = False
            if bi_on in possible_transitions:
                if self.check_agent_was_at_button(b):
                    if (row, col) == self.env_settings['button_locs'][b]:
                        l.append(bi_on)
                elif bi_on in self.my_shared_events:
                    if self.random_success(self.env_settings['thresh']):
                        l.append(bi_on)

        real_l = l[:1]
        if len(real_l) ==1: 
            for b in self.buttons: 
                abi = str(self.agent_id + 1) + b
                lbi = str(self.agent_id + 1) + 'l' + b[1:]
                bi_on = b + '_on'
                if real_l[0] == abi:
                    self.agent_was_at_button[b] = True
                elif real_l[0] == lbi:
                    self.agent_was_at_button[b] = False
                elif real_l[0] == bi_on:
                    self.flags[b] = True    

        return real_l
    
    def get_mdp_label_harder_2(self, s, s_next, u):
        possible_transitions = self.reward_machine.delta_u[u].keys()
        print(f"The transitions I can take are: {possible_transitions} .")
        l = []
        row, col = self.get_state_description(s_next)

        for b in self.buttons:
            if self.flags[b] == False:
                abi = str(self.agent_id + 1) + b
                lbi = str(self.agent_id + 1) + 'l' + b[1:]
                bi_on = b + '_on'
                if (row, col) == self.env_settings['button_locs'][b]:
                    if self.check_agent_was_at_button(b):
                        l.append(bi_on)
                    else:
                        l.append(abi)
                else:
                    if self.check_agent_was_at_button(b):
                        l.append(lbi)
                if bi_on in possible_transitions:
                    if bi_on in self.my_shared_events:
                        if self.random_success(self.env_settings['thresh']):
                                l.append(bi_on)
        
        real_l = l[:1]
        if len(real_l) == 1: 
            for b in self.buttons: 
                abi = str(self.agent_id + 1) + b
                lbi = str(self.agent_id + 1) + 'l' + b[1:]
                bi_on = b + '_on'

                if real_l[0] == abi:
                    self.agent_was_at_button[b] = True
                elif real_l[0] == lbi:
                    self.agent_was_at_button[b] = False
                elif real_l[0] == bi_on:
                    self.flags[b] = True 

        return real_l


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
        if self.flags['failed'] == True:
            return True
        else:
            return self.flags['complete']

    def reset(self):
        """ 
        Start environment over to natural state 
        """        
        self.flags= {'b1': False, 'b2': False, 'b3': False, 'b4': False, 'b5': False, 'b6': False, 'b7': False, 'b8': False, 'b9': False, 'b10': False, 'failed' : False, 'complete': False}
        self.agent_was_at_button = {'b1': False, 'b2': False, 'b3': False, 'b4': False, 'b5': False, 'b6': False, 'b7': False, 'b8': False, 'b9': False, 'b10': False, 'failed' : False} # does not start on any buttons. 


    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc))
        
        # Display the locations of the walls
        for loc in self.env_settings['walls']:
            display[loc] = -1

        for button in self.buttons:
            display[self.env_settings['button_locs'][button]] = 9

        # Display the location of the agent in the world
        row, col = self.get_state_description(s)
        display[row,col] = self.agent_id + 1

        print(display)


def play(my_shared_events, individual_events, agent_id, file_name = 'crafting_rm_1.txt'):

    base_file_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    #rm_string = os.path.join(base_file_dir,'automated_task_assignment_with_rm', 'data', 'saved_reward_machines', 'crafting_task', 'crafting_env_testing', file_name) #need to load my game 
    #print("RM STRING", rm_string)
    rm_string = os.path.join( 'data', 'saved_reward_machines', 'crafting_task', 'crafting_env_testing', file_name) #need to load my game 

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 10
    env_settings['Nc'] = 10
    env_settings['initial_states'] = [20, 23, 21]
    env_settings['walls'] = [(0, 7), (1, 7), (2, 7), (3,7), (4,7), (5,7)]
    button_locs = {'b1': (0,0), 'b2': (1,1), 'b3': (2,2), 'b4': (3,3), 'b5': (4,4), 'b6': (5,5), 'b7': (6,6), 'b8': (7,7), 'b9':(8,8), 'b10':(9,9)}
    
    env_settings['num_buttons'] = 3 # only builds this many buttons in the environment. 
    env_settings['button_locs'] = button_locs
    env_settings['thresh'] = .5
    env_settings['end_thresh'] = .5
    env_settings['p'] = 0.99

    game = Line_Of_Buttons_Env(rm_string, agent_id, env_settings, my_shared_events, individual_events)

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
            if game.flags['complete']: # Game Over # need to change my game over criteria (Change in learning?)
                break 
        else:
            print("Forbidden action")

    game.show(s)
    


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()