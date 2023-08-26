#! /usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import sys

print("TESTING")

time: float
number_of_cities: int
number_of_ants: int
number_of_iterations: int
result_path:np.ndarray = []
cost_in_time: np.ndarray = []
input_lines = sys.stdin.readlines()
time = float(input_lines[0])
number_of_cities = int(input_lines[1])
number_of_ants = int(input_lines[2])
number_of_iterations = int(input_lines[3])

for line in input_lines[4:4+number_of_cities-1]:
    result_path.append(int(line))
    

for line in input_lines[5+number_of_cities-1:5+number_of_cities-1+number_of_iterations-1]:
    cost_in_time.append(float(line))


def plot_results_time(results:np.ndarray):

    fig, ax = plt.subplots()
    ax.plot(np.arange(number_of_iterations-1),results)
    
    ax.set(xlabel='iter', ylabel='cost',
           title='Cost vs Iterations')
    
    ax.grid()

    fig.savefig("result.png")

def plot_visited_cities(path: np.ndarray):
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(number_of_cities-1),path)
    
    ax.set(xlabel='city', ylabel='step',
           title='Visited cities')
    
    ax.grid()

    fig.savefig("path.png")    

plot_results_time(cost_in_time)
plot_visited_cities(result_path)

print("DONE")