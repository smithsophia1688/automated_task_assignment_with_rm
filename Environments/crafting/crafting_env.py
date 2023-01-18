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


class CraftingEnv:

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
        self.chop_count = 0
        self.chop_thresh = 3
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file) # I do want this
        #print(self.reward_machine)

        self.agent_events_dict = {0: {'ai': 'a1', 'li': 'l1', 'tri': 'tr1'}, 1: {'ai': 'a2', 'li': 'l2', 'tri': 'tr2'}, 2: {'ai': 'a3', 'li': 'l3', 'tri': 'tr3'}}
    
        self.u = self.reward_machine.get_initial_state()
        self.last_action = -1 # Initialize last action to garbage value
        self.my_shared_events  = my_shared_events
        self.individual_events = individual_events

        self.flags = {'tree': True, 'a1': False, 'a2': False, 'a3': False, 'log': False, 'carrying': False, 'arrived': False, 'crafted': False, 'failed': False }
        #print("flags", self.flags)

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        initial_states = self.env_settings['initial_states']

        self.s_i = initial_states[self.agent_id]

        self.objects = {}
        self.objects[self.env_settings['craft_table']] = 'c' # goal location
        self.objects[self.env_settings['tree_loc']] = 't'

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
        if self.flags['carrying'] == True:
            self.env_settings['log_loc'] = (row, col)

        l = self.get_mdp_label_update_fix(s, s_next, self.u) # I think I have this correct... (yikes)
        r = 0
        tri = self.agent_events_dict[self.agent_id]['tri']
        end_thresh = self.env_settings['end_thresh']

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e) #check what this does for undefined events. TODO: 
            if u2 in self.reward_machine.T:
                if self.random_success(end_thresh):
                     self.flags['crafted'] = True
                     #print("might randomly end episode")
                     #print("You thought you were done so I let you wander for a bit then ended it. ")
            r_step = self.reward_machine.get_reward(self.u, u2) # I should be quitting here if reward is negative.
            if r_step < 0: 
                self.flags['failed'] = True
            r = r +  r_step
            # Update the reward machine state
            self.u = u2 # okay cool I get this code 
            #print(f" e is {e} and tri is {tri}")
            if e == tri:
                #print("WHY AM I NOT HERE")
                self.env_settings['log_loc'] = (row, col)
                self.flags['carrying'] = True
            if e == 'ar':
                self.flags['carrying'] = False
                self.flags['arrived'] = True
                self.flags['log'] = False
            if e == 'timber':
                self.flags['tree'] = False
                self.flags['log'] = True
            if e == 'craft':
                self.flags['crafted'] = True

        if l == []:
            if self.u in self.reward_machine.T:
                # probably never get here
                #print("might randomly end episode")
                if self.random_success(end_thresh):
                     self.flags['crafted'] = True

        return r, l, s_next


    ##############   Labeling Toolkit   ##################
    def agent_at_tree(self, row, col):
            return (row, col) == self.env_settings['tree_loc']

    def was_just_at_tree(self): 
        ai = self.agent_events_dict[self.agent_id]['ai']
        return self.flags[ai]

    def carrying_responsibility(self):
        tri = self.agent_events_dict[self.agent_id]['tri']
        if tri in self.individual_events:
            #print("I have carrying responsibility")
            return True
        #print("I do not have carrying responsibility")
        return False #IT would be an accident to do this. cannot be a shared event.

    def cutting_responsibility(self):
        #print("i am looking for ", self.agent_events_dict[self.agent_id]['ai'])
        if self.agent_events_dict[self.agent_id]['ai'] in self.individual_events:
            #print("I have cutting responsibility")
            return True # it was assigned to me during task assignment. 
        #print("I do not have cutting responsibility") 
        return False

    def crafting_responsibility(self):
        if 'craft' in self.individual_events:
            #print(" I have crafting responsibility")
            return True
        #print("I do not have crafting responsibility")
        return False

    def try_timber(self, l, thresh): #NOT USED
        '''
        I could be in trouble when:
            - if I have cutting responsibility but do not have timber (would unfairly give timber)
            - if I did not have cutting responsibility and  ONLY I had timber and literally no one else '''
        
        if self.cutting_responsibility():
            if self.was_just_at_tree():
                if 'timber' in self.my_shared_events:
                    print("I did my part, simulate timber")
                    if np.random.random() <= thresh: # If you are expecting arrival to be shared. 
                        print("... randomly cut a tree wow")
                        l.append('timber' )
                        self.flags['tree'] = False
                        self.flags['log'] = True
                else:
                    print("I did my part and I know it, tree falls")
                    if 'timber' not in self.individual_events:
                        print("ALERT YOU DID NOT EXPECT THIS TO HAPPEN ")
                        raise ValueError("This is a problem, cutting responsibility but no timber")
                    l.append('timber')
            else:
                print("not ready for tree to fall, didn't do my part yet")
                print("THIS SHOULD NEVER HAPPEN?")
        else: 
            if 'timber' in self.my_shared_events: #could put 'my events' 
                print("not my job to cut, simulate timber")
                if np.random.random() <= thresh: # If you are expecting arrival to be shared. 
                    l.append('timber' )
                    self.flags['tree'] = False
                    self.flags['log'] = True
            else: 
                print("It was not my job to cut and I do not care if timber happends. Just move to log")
                if np.random.random() <= thresh: # If you are expecting arrival to be shared. 
                    self.flags['tree'] = False
                    self.flags['log'] = True
                
    def try_log_arrival(self, l, row, col, thresh): #NOT USED 
        if self.carrying_responsibility():
            if self.flags['carrying'] == True:
                if (row, col) == self.env_settings['craft_table']:
                    self.flags['arrived'] = True
                    self.flags['log'] = False
                    self.flags['carrying'] = False
                    l.append('ar')
                    #print(" It is my job to carry the log and I just arrived at the table and I was already carrying the log")
                else:
                    print("Its my job to carry and I am carrying but am not at table yet")
            else:
                print("its my job to carry but I have not picked it up yet")
        else:
            if 'ar' in self.my_shared_events:
                print("I am expecting someone else to bring the log and I await their random arrival ")
                if np.random.random() <= thresh: 
                    self.flags['arrived'] = True
                    self.flags['carrying']  = False
                    self.flags['log'] = False
                    l.append('ar')
            else: 
                print("I could not care less if the log arrived. I just need to learn no not pick it up")
                if np.random.random() <= thresh: 
                    self.flags['arrived'] = True
                    self.flags['carrying']  = False
                    self.flags['log'] = False
    
    def try_crafting(self, l, row, col, thresh): # NOT USED
        if self.crafting_responsibility():
            if (row, col) == self.env_settings['craft_table']:
                l.append('craft')
                self.flags['crafted'] = True
            else:
                print("it is my job to craft but I have not yet done that")
        else:
            print("It is not my job to craft, randomly simulate end of task")
            if np.random.random() <= thresh: 
                self.flags['crafted'] = True
    
    
    ##################   Labeling   ######################  
    
    def get_mdp_label_tree(self, s_next, thresh):
        '''
        I have mdp label for tree section
        2 assumptions I am making: 
        1) I am expecting that if the agent has cutting responsibility, then they also have "timber" 
        2) I am expecting that if an agent did not have cutting responsibility and has "timber", than some else has "timber" 
        If either of these are wrong, that is a suprise to me. 
        I feel quite confident about the 2nd one. 
        I feel less confident about the 1st one (will raise a value error)

        '''
        row, col = self.get_state_description(s_next)
        l = []


        ai = self.agent_events_dict[self.agent_id]['ai']
        li = self.agent_events_dict[self.agent_id]['li']

        if self.was_just_at_tree(): #I was just at the tree
            if self.agent_at_tree(row,col):
                #print("I was previously at the tree and I am still here, won't add another ai")
                self.flags[ai] = True
                if self.cutting_responsibility():
                    if 'timber' in self.my_shared_events:
                        #print("It is my job to cut down the tree and I did my part. Randomly simulate tree")
                        if np.random.random() <= thresh: # If you are expecting arrival to be shared. 
                            l.append('timber' )
                            self.flags['tree'] = False
                            self.flags['log'] = True
                    else:
                        #print("I did my part of arriving at the tree and it is not shared withanyone so I will auto cut it down. ")
                        l.append('timber')
                        if 'timber' not in self.individual_events:
                            #print("ALERT YOU DID NOT EXPECT THIS TO HAPPEN ")
                            raise ValueError("This is a problem, cutting responsibility but no timber")                   
                else: # not repsonsible 
                    #print(" I was already at the tree but I am not responsible of cutting down the tree but should have already had negative reward")
                    raise ValueError("You should have already accumulated a negative reward from previous arrival ")

            else:
                #print(" I am not currently at the tree even though I was just there")
                l.append(li)
                self.flags[ai] = False
                if not self.cutting_responsibility():
                    raise ValueError("You should have already accumulated a negative reward from previous arrival ")
                #else:
                #    print(" It was my job to cut down the tree but I left")
        else: #was not just at tree
            #print("I was not just at the tree so I cannot make the tree fall ")
            if self.agent_at_tree(row,col):
                #print("I just arrived at the tree for anew the first time. add ai ")
                l.append(ai)
                self.flags[ai] = True
            else:
                if not self.cutting_responsibility():
                    #print("I was not previously at the tree, but I am at the tree now and it is not my job to cut down the tree")
                    if np.random.random() <= thresh: # If you are expecting arrival to be shared. 
                        #print("... randomly cut a tree wow")
                        if 'timber' in self.my_shared_events:
                            l.append('timber')
                        self.flags['tree'] = False
                        self.flags['log'] = True
                #else: 
                #    print("I am not at the tree, was not previously at the tree, and am responsible for cuttin git down. Wait for me!")
                #print(" wow I'm still not doing much")
        
        return l

    def get_mdp_label_log(self, s_next, thresh):
        row, col = self.get_state_description(s_next)
        l = []

        tri = self.agent_events_dict[self.agent_id]['tri']

        if self.flags['carrying'] == True:
            
            self.env_settings['log_loc'] = (row, col)
            if (row, col) == self.env_settings['craft_table']:
                #print(" The log is being carried and I am at the craft table ")
                l.append('ar')
                self.flags['carrying'] = False
                self.flags['arrived'] = True
                self.flags['log'] = False
            else:
                #print(" the log is being carried and I am not at the craft table yet")
                l.append(tri)

        else: # not carrying log yet
            if (row, col) == self.env_settings['log_loc']:
                #print("I was not carrying the log yet but I just arrived at the logs location. Picking it up now")
                self.flags['carrying'] = True
                l.append(tri)
            else: # not carrying and not there yet
                if not self.carrying_responsibility():
                    #print(" I am not carrying the log yet and I am not at the log location. It is also not my responsibility. Randomly simulate arrival to table")
                    if np.random.random() <= thresh:
                        #if 'ar' in self.individual_events: # You don't need this since ar cannot be an accident 
                        l.append('ar')
                        self.flags['carrying'] = False
                        self.flags['arrived'] = True
                        self.flags['log'] = False
        #print("leaving", self.flags)   
        return l

    def get_mdp_label_craft(self, s_next, thresh):
        '''
        Note: if it is not the correct agent to craft, they will never craft
        '''
        row, col = self.get_state_description(s_next)
        l = []
        if self.crafting_responsibility():
            if (row, col) == self.env_settings['craft_table']:
                l.append('craft')
                self.flags['crafted'] = True
            #else:
            #    #print("it is my job to craft but I have not yet done that")
        else:
            #print("It is not my job to craft, randomly simulate end of task")
            #if np.random.random() <= thresh: 
            self.flags['crafted'] = True

        return l

    def get_mdp_label(self, s, s_next, u):
        tree_thresh = .8
        log_thresh = .8
        craft_thresh = .8
        if self.flags['tree'] == True:
            #print(" we have a tree")
            l = self.get_mdp_label_tree(s_next, tree_thresh)
        elif self.flags['log'] == True:
            #print(" we have a log")
            l = self.get_mdp_label_log(s_next, log_thresh)
        elif self.flags['arrived'] == True:
            #print("crafting time")
            l = self.get_mdp_label_craft(s_next, craft_thresh)

        return l 
   
    def random_success(self, thresh):
        if np.random.random() <= thresh:
            return True
        return False

    def get_mdp_label_update(self, s, s_next, u):
        # this cannot be a function of the name of the reward machine. It needs to be based off of tranistions
        possible_transitions = self.reward_machine.delta_u[u].keys()
        print(f"The transitions I can take are: {possible_transitions} .")
        
        ai = self.agent_events_dict[self.agent_id]['ai']
        li = self.agent_events_dict[self.agent_id]['li']
        tri = self.agent_events_dict[self.agent_id]['tri']
        #print(f'ai is {ai}, li is {li}, tri is {tri}. ')
        thresh = .3
        l = []
        row, col = self.get_state_description(s_next)

        for e in possible_transitions: # at the mercy of what order e gets pulled (bad)
            if e == 'timber':
                if e in self.my_shared_events:
                    print(f"try simulating {e}")
                    if self.random_success(thresh):
                        l.append(e)
                else:
                    print(f"{e} is not shared I will have to do it myself")
                    print(f"Can {e} ever exist not shared? ")
            
            elif e == ai:
                if e in self.my_shared_events:
                    print(f"try simulating {e}")
                    if self.random_success(thresh):
                        l.append(e)
                else:
                    print(f"{e} is not shared I will have to do it myself")
                    if (row, col) == self.env_settings['tree_loc']:
                        l.append(e)
            
            elif e == li:
                if e in self.my_shared_events:
                    print(f"try simulating {e}")
                    if self.random_success(thresh):
                        l.append(e)
                else:
                    print(f"{e} is not shared I will have to do it myself")
                    if (row, col) != self.env_settings['tree_loc']:
                        l.append(e)
        
            elif e == tri: # if this is an accident it will always be in my possible transitions. Maybe I should be looking at this all differently 
                if e in self.my_shared_events:
                    print(f"try simulating {e}")
                    if self.random_success(thresh):
                        l.append(e)
                else:
                    if (row, col) == self.env_settings['log_loc']:
                        l.append(e) 
                    print(f"{e} is not shared I will have to do it myself")
            elif e == 'ar':
                if e in self.my_shared_events:
                    print(f"try simulating {e}")
                    if self.random_success(thresh):
                        l.append(e)
                else:
                    print(f"{e} is not shared I will have to do it myself")
                    if (row, col) == self.env_settings['craft_table']:
                        l.append(e)
            
            elif e == 'craft':
                if e in self.my_shared_events:
                    print(f"try simulating {e}")
                    if self.random_success(thresh):
                        l.append(e)
                else:
                    print(f"{e} is not shared I will have to do it myself")
                    if (row, col) == self.env_settings['craft_table']:
                        l.append(e)
            #else:
            #    print(f"that was weird, how did {self.agent_id} get {e}?")
        
        return l[:1] #only return the first successful event. 

    def get_mdp_label_update_fix(self, s, s_next, u):
        if u in self.reward_machine.delta_u.keys():
            possible_transitions = self.reward_machine.delta_u[u].keys()
        else:
            possible_transitions = []
        #print(f"The transitions I can take are: {possible_transitions} .")

        ai = self.agent_events_dict[self.agent_id]['ai']
        li = self.agent_events_dict[self.agent_id]['li']
        tri = self.agent_events_dict[self.agent_id]['tri']

        thresh = self.env_settings['thresh'] 
        l = []
        row, col = self.get_state_description(s_next)

        # What order do I check:
        # li
        if li in possible_transitions:
            # only happens when I have already arrived (NOT TRUE: could be an accident to leave always. ) UPDATE: now true
            if (row, col) != self.env_settings['tree_loc']:
                l.append(li)
        if 'timber' in possible_transitions:
            # only happens when the tree is ready to fall
            if 'timber' in self.my_shared_events:
                #print(f"try simulating 'timber'")
                if self.random_success(thresh):
                    l.append('timber')
            #else:
            #    print(" the combo of timber being possible (means its either an accident or my event) but not being shared shouldn't happen, right?")
            #    # you could automatically add but it could mess up, someone could get stuck here? 
        
        
        if ai in possible_transitions:
            if (row, col) == self.env_settings['tree_loc']:
                l.append(ai)
        if 'ar' in possible_transitions:
            if self.flags['carrying'] == False:
                # I am not carring the log yet
                if 'ar' in self.my_shared_events:
                    if self.random_success(thresh):
                        #print("Randomly arrived sucess")
                        l.append('ar')
            else:
                if (row, col) == self.env_settings['craft_table']:
                    l.append('ar')
        if tri in possible_transitions:
            if (row, col) == self.env_settings['log_loc']:
                l.append(tri)
        
        if 'craft' in possible_transitions:
            if (row, col) == self.env_settings['craft_table']:
                l.append('craft')

        # I want to get rid of the next few lines:
        #if self.flags['arrived'] == True:
        #    if 'craft' not in self.individual_events:
        #        l.append('craft')

        #print(f"full l is {l}")

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
        if self.flags['crafted'] == True:
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
        display = np.zeros((self.Nr, self.Nc))
        
        # Display the locations of the walls
        for loc in self.env_settings['walls']:
            display[loc] = -1

        display[self.env_settings['craft_table']] = 9
        #pre cut tree: 
        if self.flags['tree']:
            display[self.env_settings['tree_loc']] = 8

        # Need a way to track where the log is... and display..
        if self.flags['log']: 
            display[self.env_settings['log_loc']] = 7

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
    env_settings['initial_states'] = [20, 81, 99]
    env_settings['walls'] = [(0, 7), (1, 7), (2, 7), (3,7), (4,7), (5,7), (8,7), (9,7)]
    env_settings['craft_table'] = (1,9)
    env_settings['tree_loc'] = (0, 3)
    env_settings['log_loc'] = (0, 4)

    env_settings['thresh'] =.5
    env_settings['end_thresh'] = .5
    env_settings['p'] = 0.99

    game = CraftingEnv(rm_string, agent_id, env_settings, my_shared_events, individual_events)
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
            print("log locations:", game.env_settings['log_loc'])
            print("Label: ", l)
            print("Reward: ", r)
            print("RM state: ", game.u)
            print("task flags: ", game.flags)
            print('full trace:', full_l)
            print("---------------------")

            if game.flags['failed']:
                break
            if game.flags['crafted']: # Game Over # need to change my game over criteria (Change in learning?)
                break 
        else:
            print("Forbidden action")

    game.show(s)
    


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()