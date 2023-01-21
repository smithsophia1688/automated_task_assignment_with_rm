import pickle
import matplotlib.pyplot as plt
import numpy as np

# run these on the line experiment, since it will allow for different paths? 

def plot_multi_agent_results(ax,best_tester, num_agents, label="zoe",color="red"):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps = list()

    plot_dict_1 = best_tester.results['testing_steps']
   
    for step in plot_dict_1.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_1[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict_1[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict_1[step]), 75))
            current_step.append(sum(plot_dict_1[step])/len(plot_dict_1[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_1[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_1[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_1[step]),75))
            current_step.append(sum(plot_dict_1[step])/len(plot_dict_1[step]))

        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps.append(step)

    
    ax.plot(steps, prc_25, alpha=0)
    ax.plot(steps, prc_50, color=color, label=label)
    ax.plot(steps, prc_75, alpha=0)
    ax.grid()
    ax.fill_between(steps, prc_50, prc_25, color=color, alpha=0.25)
    ax.fill_between(steps, prc_50, prc_75, color=color, alpha=0.25)
    ax.set_ylabel('Testing Steps to Task Completion', fontsize=15)
    ax.set_xlabel('Training Steps', fontsize=15)
    ax.locator_params(axis='x', nbins=5)

se_file_name = 'data/saved_testers/crafting_task/2023_01_18-10.19.58_PM_1_0_0.pkl'
f_file_name = 'data/saved_testers/crafting_task/2023_01_18-11.26.23_PM_0_1_0.pkl'
u_file_name = 'data/saved_testers/crafting_task/2023_01_19-11.27.45_AM_0_0_1.pkl'
u2_file_name = 'data/saved_testers/crafting_task/2023_01_19-02.24.52_PM_0_0_1.pkl'
mixed_file_name = 'data/saved_testers/crafting_task/2023_01_19-01.18.30_PM_0.5_0.5_0.5.pkl'


five_se_file_name = 'data/saved_testers/five_agent_task/2023_01_20-12.25.41_AM_1_0_0.pkl'
five_f_file_name = 'data/saved_testers/five_agent_task/2023_01_20-12.27.45_AM_0_1_0.pkl'
five_u_file_name = 'data/saved_testers/five_agent_task/2023_01_20-12.29.54_AM_0_0_1.pkl'
five_u2_file_name = 'data/saved_testers/five_agent_task/2023_01_20-10.10.51_AM_0_0_1.pkl'
five_mixed_file_name = 'data/saved_testers/five_agent_task/2023_01_20-12.32.05_AM_0.5_0.5_0.5.pkl'

exp = 'crafting'
# exp = 'five'
if exp == 'crafting':
    num_agents = 3
    with open(se_file_name , 'rb') as f :
        se_tester = pickle.load(f)
    print("opened 1")
    with open(f_file_name , 'rb') as f :
        f_tester = pickle.load(f)
    print("opened 2")

    with open(u_file_name , 'rb') as f :
        u_tester = pickle.load(f)
    print("opened 3")
    with open(mixed_file_name , 'rb') as f :
        m_tester = pickle.load(f)
    print("opended 4")

    with open(u2_file_name , 'rb') as f :
        u2_tester = pickle.load(f)
    print("opended 5")

else:
    num_agents = 5
    with open(five_se_file_name , 'rb') as f :
        se_tester = pickle.load(f)
    print("opened 1")
    with open(five_f_file_name , 'rb') as f :
        f_tester = pickle.load(f)
    print("opened 2")

    with open(five_u_file_name , 'rb') as f :
        u_tester = pickle.load(f)
    print("opened 3")
    with open(five_mixed_file_name , 'rb') as f :
        m_tester = pickle.load(f)
    print("opended 4")

    with open(five_u2_file_name , 'rb') as f :
        u2_tester = pickle.load(f)
        
    # u2_tester = None
    print("opended 5")
    

print("all opened")

fig, ax = plt.subplots()
plot_multi_agent_results(ax, se_tester, num_agents,label='shared events', color= 'red')
plot_multi_agent_results(ax, f_tester, num_agents, label = 'fairness', color = 'blue')
plot_multi_agent_results(ax, u_tester, num_agents, label = 'utility', color = 'green')
plot_multi_agent_results(ax, m_tester, num_agents, label = '[.5,.5,.5]', color = 'purple')
if u2_tester != None:
    plot_multi_agent_results(ax, u2_tester, num_agents, label = 'utility take 2', color = 'orange')
plt.legend(loc = 'upper right')
plt.show()
# import tikzplotlib
# tikzplotlib.save("data/saved_plots/test_scores.tex")


print("I am here")