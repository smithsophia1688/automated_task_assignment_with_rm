import pickle
import matplotlib.pyplot as plt
import numpy as np


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
    
best_file_name = '2022_11_27-01.09.41_PM_best.pkl'
best_full_file_name = 'data/saved_testers/crafting_task/' + best_file_name 

trivial_file_name = '2022_11_27-03.50.41_PM_trivial.pkl'
trivial_full_file_name = 'data/saved_testers/crafting_task/' + trivial_file_name 

with open(best_full_file_name , 'rb') as f :
    best_tester = pickle.load(f)

with open(trivial_full_file_name , 'rb') as f :
    trivial_tester = pickle.load(f)

print(type(best_tester))
print(type(trivial_tester))
num_agents = 3

fig, ax = plt.subplots()
plot_multi_agent_results(ax, best_tester, num_agents,label='best', color= 'red')
plot_multi_agent_results(ax, trivial_tester, num_agents, label = 'trivial', color = 'blue')
plt.legend()

import tikzplotlib
tikzplotlib.save("data/saved_plots/test_3.tex")


print("I am here")