# General useful imports
import numpy as np
from numpy import arange,linspace,mean, var, std
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from numpy.random import seed, random, randint, uniform, choice, binomial, geometric, poisson, exponential, normal
import math
from collections import Counter
import pandas as pd
from scipy.stats import norm,expon,poisson,binom,bernoulli


# Round to 4 decimal places
def round4(x):
    return np.around(x,4)


# Queue Simulation Code

# The following code simulates a simple queue and server, in which tasks arrive at a FIFO queue and wait for service; after they complete their service, they leave the system.
#
#
# Tasks are defined by two parameters, their service time (how long they need service) and their arrival time. The simulation is entirely determined by the list of tasks and their parameters. For this problem, several lists of tasks are given explicitly, and the first thing to do is to try each of these and understand what information is being output at the end:
#
# A Gannt Chart showing over time what happens to each of the tasks as they arrive, wait in the queue, and get served.
# Chart of CPU Utilization over time
# Chart of the Queue Length over time
# Distribution of the Queue Length (as a random variable)
# Various quantitative measures from the simulation, namely,
# totalTime -- just the time in seconds for the whole simulation
# meanServiceTime -- the average over all tasks of the time they required for service (this was specified in the original task list, but the statistic is the mean over all tasks)
# meanInterarrivalTime -- the average over all tasks of the intervals between arrivals
# meanWaitTime -- the average over all tasks of the amount of time they spend waiting in the queue
# cpuUtilization -- the percentage of time the server (CPU) is busy
# meanQueueLength -- the expected value (mean) of the random variable representing the length of the queue over time
# stdQueueLength -- the standard deviation of the random variable representing the length of the queue over time



# task in list is [arrival time, service request, cpu start time]

# We assume in this simulation that no tasks arrive at the same time; you must specify the
# first two components, and the last is filled in by the simulation


def runSimulation(taskList):

    task = taskList.copy()

    front = 0
    next = 0          # when front == next, Q is empty
    cpu = -1          # when cpu == -1, CPU is empty

    t = 0       # clock

    #    cpuUtil = []         # CPU utilization
    #    Q = []               # distribution of Q size

    while(front < len(task) or cpu != -1):
        # beginning of this time slice

        # if there is a next task which enters Q now, do so
        if(next < len(task) and task[next][0] == t):
            next += 1
        # if CPU is empty and Q is not, move front of queue to CPU
        if(cpu == -1 and front != next):
            cpu = front             # move front of Q to cpu
            front += 1              # move front of Q to next task
            task[cpu][2] = t       # store cpu start time in task


        # increment clock to next event
        # could be task arrival in Q or task finish in cpu

        nextCpuFinish = (task[cpu][1]+task[cpu][2])
        if(next < len(task) and cpu != -1):
            if(task[next][0] < nextCpuFinish):
                t = task[next][0]               # next is arrival of new task in Q
            else:
                t = nextCpuFinish                # next is cpu finish
                cpu = -1
        elif(next < len(task)):
            t = task[next][0]               # next is arrival of new task in Q
        elif(cpu != -1):
            t = nextCpuFinish                # last job has finished

        # end of this time slice

        # if task in CPU has finished, remove it
        if(nextCpuFinish == t):
            cpu = -1


    #    print("\nFinished Tasks: " + str(task))

    # return completed task list and statistics
    # []
    return task


def E(X):
    (R,P) = X
    return sum(p*q for p,q in zip(R,P))

def Var(X):
    (R,P) = X
    mu = E(X)
    return E( ([(x-mu)**2 for x in R], P ))

def stdev(X):
    return (Var(X)**0.5)

# Print out list of tasks in nice format

def pprint(tasks):
    if len(tasks) == 0:
        print("Task list is empty.")
        return
    if(sum([w for [a,s,w] in tasks]) == 0):
        print("Task #\tStart\tService")
        print('0\t' +str(round4(tasks[0][0]))+'\t'+str(round4(tasks[0][1])))
    else:
        print("Task #\tStart\tService\tWait")
        print('0\t' +str(round4(tasks[0][0]))+'\t'+str(round4(tasks[0][1]))+'\t'+str(round4(tasks[0][2])))
    for k in range(1,len(tasks)):
        if(tasks[k][2] == 0):
            print(str(k)+'\t'+str(round4(tasks[k][0]))+'\t'+str(round4(tasks[k][1])))
        else:
            wait_time = tasks[k][2] - tasks[k][0]
            print(str(k)+'\t'+str(round4(tasks[k][0]))+'\t'+str(round4(tasks[k][1]))+'\t'+str(wait_time))


def drawQDistribution(X, P):

    # This is the boundaries of the bins, they are half way between each
    #    of the discrete points in the sample set
    if(len(X) > 5):
        plt.figure(figsize=(15,3))

    bins = [x-0.5 for x in range(min(X),max(X)+2)]

    plt.hist(X, bins, weights=P, ec='k')
    plt.xticks(X)
    plt.title("Distribution X of Queue Lengths")
    plt.ylabel("Probability")
    plt.xlabel("Queue Lengths")
    plt.show()

    print("E[X]     = {0:.4f}".format(round(E((X,P)),4)))
    print("Var(X)   = {0:.4f}".format(round(Var((X,P)),4)))
    print("stdev(X) = {0:.4f}".format(round(stdev((X,P)),4)))


# this prints out the Gannt Chart and also prints out the
# probability distribution of the queue length

def displayResults(task):

    # print("\nCompleted task list:" + str(task))
    print("\nNumber of Tasks: " + str(len(task)))
    totalTime = task[-1][2]+task[-1][1]
    print("\nTotal Time: " + str(round4(totalTime)))
    print("\n")

    # Print GANTT Chart

    fig = plt.figure(figsize=(15,10))
    fig.subplots_adjust(hspace=.5)
    ax1 = fig.add_subplot(311)
    plt.yticks(range(len(task)))
    plt.ylim((-0.5,len(task)))
    plt.title('GANNT Chart')
    plt.ylabel('Task Number')
    plt.xlabel('Time Slot')

    for k in range(len(task)):
        plt.hlines(k, task[k][0], task[k][2], color='C0',linestyle='dotted',linewidth=4)
        plt.hlines(k, task[k][2], task[k][1]+task[k][2], color='C0',linestyle='solid',linewidth=4)

        # Determine means of various parameters

    sumService = 0
    for i in range(len(task)):
        sumService += task[i][1]

    meanServiceTime = sumService / len(task)

    sumInterarrival = 0
    for i in range(1,len(task)):
        sumInterarrival += task[i][0] - task[i-1][0]

    meanInterarrivalTime =  sumInterarrival / len(task)

    sumWait = 0
    for i in range(len(task)):
        sumWait += task[i][2] - task[i][0]

    meanWaitTime = sumWait / len(task)

    sumService = 0
    for i in range(len(task)):
        sumService += task[i][1]

    cpuUtilization = sumService / totalTime

    # Now figure out CPU utilization in each time slot

    # first figure out all possible cpu event times

    bins = []

    for k in range(len(task)):
        bins.append(task[k][2])
        bins.append(task[k][1]+task[k][2])

    bins = list(set(bins))
    bins.sort()

    X = []
    for i in range(len(task)):
        X.append(task[i][2])


        # Display CPU Utilization Chart

    fig.add_subplot(312,sharex=ax1)
    plt.hist(X,bins,ec='k')
    plt.title('CPU Utilization')
    plt.ylabel('Usage')
    plt.xlabel('Time Slot')
    plt.ylim((0,1.25))

    # Calculate Queue Length Distribution over time slots and by length

    bins = []

    for k in range(len(task)):
        bins.append(task[k][0])        # only add times when q changes
        bins.append(task[k][2])

    bins = list(set(bins))
    bins.sort()


    X = []      # X will have bin start times, repeated for each arrival in queue

    totalTime = task[-1][1]+task[-1][2]

    binProbabilities = [(bins[i]-bins[i-1])/totalTime for i in range(1,len(bins))]
    #    # add last probability for last bin (when last task in cpu)
    binProbabilities.append(1-sum(binProbabilities))
    binCount = [0]*(len(bins))     # how many tasks in queue in each bin

    for k in range(len(bins)):
        for i in range(len(task)):
            if(task[i][0] <= bins[k] < task[i][2]):
                binCount[k] += 1
                X.append(bins[k])

            #    print(X)
    #    print(binCount)

    fig.add_subplot(313,sharex=ax1)
    plt.hist(X,bins,ec='k')
    plt.title('Queue Length over Time')
    plt.ylabel('Length of Queue')
    plt.xlabel('Time Slot')
    plt.show()

    # Now calculate the probability distribution of the queue lengths
    limit = 100
    length = min(limit,max(binCount))+1
    P = [0]*length   # P will be probability distribution for range 0 .. 99

    for k in range(len(bins)):
        if(binCount[k] < limit):
            P[binCount[k]] += binProbabilities[k]

    drawQDistribution(range(length),P)
    print("\nMean Service Time: " + str(round4(meanServiceTime)))
    print("\nMean Arrival Rate: " + str(round4(1/meanInterarrivalTime)))
    print("\nMean Inter-arrival Time: " + str(round4(meanInterarrivalTime)))
    print("\nCPU Utilization: " + str(round4(cpuUtilization)))
    print("\nMean Wait Time: " + str(round4(meanWaitTime)))


# Return a pair (X,P) which is the distribution of the queue lengths
def getQueueDistribution(task):

    bins = []

    for k in range(len(task)):
        bins.append(task[k][0])        # only add times when q changes
        bins.append(task[k][2])

    bins = list(set(bins))
    bins.sort()

    X = []      # X will have bin start times, repeated for each arrival in queue

    totalTime = task[-1][1]+task[-1][2]

    binProbabilities = [(bins[i]-bins[i-1])/totalTime for i in range(1,len(bins))]
    #    # add last probability for last bin (when last task in cpu)
    binProbabilities.append(1-sum(binProbabilities))
    binCount = [0]*(len(bins))     # how many tasks in queue in each bin

    for k in range(len(bins)):
        for i in range(len(task)):
            if(task[i][0] <= bins[k] < task[i][2]):
                binCount[k] += 1
                X.append(bins[k])



            # Now calculate the probability distribution of the queue lengths
    limit = 100
    length = min(limit,max(binCount))+1
    P = [0]*length   # P will be probability distribution for range 0 .. 99

    for k in range(len(bins)):
        if(binCount[k] < limit):
            P[binCount[k]] += binProbabilities[k]

    return ([x for x in range(length)],P)

def analyzeResults(task):

    totalTime = task[-1][2]+task[-1][1]


    sumService = 0
    for i in range(len(task)):
        sumService += task[i][1]

    meanServiceTime = sumService / len(task)

    sumInterarrival = 0
    for i in range(1,len(task)):
        sumInterarrival += task[i][0] - task[i-1][0]

    meanInterarrivalTime =  sumInterarrival / len(task)

    sumWait = 0
    for i in range(len(task)):
        sumWait += task[i][2] - task[i][0]

    meanWaitTime = sumWait / len(task)

    sumService = 0
    for i in range(len(task)):
        sumService += task[i][1]

    cpuUtilization = sumService / totalTime

    # Now figure out CPU utilization in each time slot

    # first figure out all possible cpu event times

    bins = []

    for k in range(len(task)):
        bins.append(task[k][2])
        bins.append(task[k][1]+task[k][2])

    bins = list(set(bins))
    bins.sort()

    X = []
    for i in range(len(task)):
        X.append(task[i][2])



        # Calculate Queue Length Distribution over time slots and by length

    bins = []

    for k in range(len(task)):
        bins.append(task[k][0])        # only add times when q changes
        bins.append(task[k][2])

    bins = list(set(bins))
    bins.sort()


    X = []      # X will have bin start times, repeated for each arrival in queue

    totalTime = task[-1][1]+task[-1][2]

    binProbabilities = [(bins[i]-bins[i-1])/totalTime for i in range(1,len(bins))]
    #    # add last probability for last bin (when last task in cpu)
    binProbabilities.append(1-sum(binProbabilities))
    binCount = [0]*(len(bins))     # how many tasks in queue in each bin

    for k in range(len(bins)):
        for i in range(len(task)):
            if(task[i][0] <= bins[k] < task[i][2]):
                binCount[k] += 1
                X.append(bins[k])


            # Now calculate the probability distribution of the queue lengths
    limit = 100
    length = min(limit,max(binCount))+1
    P = [0]*length   # P will be probability distribution for range 0 .. 99

    for k in range(len(bins)):
        if(binCount[k] < limit):
            P[binCount[k]] += binProbabilities[k]

    X = range(length)

    meanQueueLength = E((X,P))

    stdQueueLength = stdev((X,P))


    return (totalTime, meanServiceTime, meanInterarrivalTime, meanWaitTime, cpuUtilization,meanQueueLength,stdQueueLength)



# Run each of the example task lists ex0, ..., ex3 in the following code cell to see how they behave.

# Some simple examples of task lists to explore the basic ideas.


# Simulation will fill in last component in task


ex0 = [[0,1,0], [2,2,0], [3,1,0], [4,3,0], [5,5,0]]

ex1 = [[0, 1.0, 0], [0.5, 0.3, 0], [1.4, 1, 0], [1.9, 0.25, 0], [2.0, 0.5, 0]]

ex2 = [[0,2.1,0], [1.9,2.34,0], [3.56,4.1,0], [4.12,3.4,0], [5.2,2.43,0]]

ex3 = [[0,2,0], [1.1,3,0], [3.4,4,0], [3.45,3,0], [5,3.23,0],[6.2,2.9,0],[6.4,2.32,0], [9.99,1.2,0], [10.3,3.4,0], [12.8,3.9,0], [15.2,3.4,0],
       [15.67,2.43,0],[17.01,2.8,0],[18.8,2.2,0],[20.1,2.99,0],[21.7,5.34,0],[24.4,2.2,0]]

def displaySimulation(ex):
    print("\nSimulation Running ... \n")
    if len(ex) <= 20:
        print("Input Task List: " + str(ex)+'\n')
    completedTasks = runSimulation(ex)

    # The following will display various charts and print out the stats; this is just for illustration and
    # debugging
    print("\n-------------------------------------------------------------------------------------")
    print("\nHere is a display of the results of the simulation for illustration and debugging:\n")
    displayResults(completedTasks)

    # the following will not display anything, but just return the statistics
    (totalTime, meanServiceTime, meanInterarrivalTime, meanWaitTime, cpuUtilization,meanQueueLength,stdQueueLength) = analyzeResults(completedTasks)
    print("\n-------------------------------------------------------------------------------------")
    print("\nHere is a display of the statistics collected by analyzeResults(...):\n")
    print("\nCompleted Task List:")
    pprint(completedTasks)
    print("\n(totalTime, meanServiceTime, meanInterarrivalTime, meanWaitTime, cpuUtilization, meanQueueLength, stdQueueLength)\n")
    print((round4(totalTime), round4(meanServiceTime), round4(meanInterarrivalTime), round4(meanWaitTime), round4(cpuUtilization), round4(meanQueueLength), round4(stdQueueLength)))



displaySimulation( ex0 )      # <<== change this here



# Producing Exponentially Distributed Input Task Lists
# This function which will create a task list of N tasks, with exponentially-distributed arrival times and service times. The three parameters will be
#
# - N = how many tasks will enter the system
# - mst = the MEAN service time in seconds
# - lam = the MEAN arrival rate in arrivals / second
# and there is a 4th, derived, parameter:
#
# - beta = 1 / lam = the MEAN inter-arrival time in seconds.

# Return a list of tasks of length N, with mean arrival rate lam and
# mean service time mst. Remember that `expon.rvs` takes the mean beta
# not lam.  See first cell for syntax.

def getTaskList(num_tasks,lam,mst):
    tasks = [[0,np.around(expon.rvs(0,mst),4),0]]

    for i in range(1,num_tasks):
        tasks.append([])
        tasks[i].append(np.around(tasks[i-1][0] + expon.rvs(0,1/lam),4))
        tasks[i].append(np.around(expon.rvs(0,mst),4))
        tasks[i].append(0)

    return tasks

# Test

seed(0)

# Uncomment to test

for i in getTaskList(10,1.0,0.6):
    print(i)



# Run the experiments, should be something like this
# Start should INCREASE, with difference with mean around 1,
# Service time should have mean around 0.6, and DOES NOT increase.

'''Task #	Start	Service Time
0	0	0.4775
1	1.2559	0.5539
2	2.0431	0.3306
3	3.0813	0.3453
4	5.3048	1.9889
5	5.7884	0.9413
6	6.5411	0.5037
7	9.1393	0.0442
8	9.2305	0.0123
9	11.018	0.9035
'''
print()


# Change this line for number of tasks, 30 should be fine for this problem.
num_tasks = 30              # Number of tasks

# LEAVE THIS LINE THE SAME
mst = 0.5        # Mean service time, service time is X ~ Exp( mst ) in units of seconds

# Experiment by changing lam to below and above 2.0

lam = 2           # Mean rate of arrivals per second

seed(0)

print("\nRunning Simulation with mean arrival rate = " + str(lam) + " and mean service time " + str(mst) + " ... ")

TS = getTaskList(num_tasks,lam,mst)

displaySimulation( TS )



# Analyzing the CPU Utilization

# Perform an experiment to analyze the CPU Utilization.
# Run the simulation num_trials= 104  (or more) times, recording the CPU utilization for each trial,
# and then plot the distribution of this data and report the mean value as a confidence interval.

num_trials = 10**5
# Complete the following stub and simply print out the result shown

seed(0)

# Run num_trial simulation run, and collect
# a list of the kth statistic in the tuple returned by analyzeResults.
# num_tasks, beta, and mst describe the initial task list

def getStat(k,num_tasks,lam,mst,num_trials):
    result = []

    for i in range(num_trials):
        result.append(analyzeResults( runSimulation( getTaskList(num_tasks,lam,mst) ) )[k])

    return result

num_tasks = 100
lam = 1.0
mst = 0.7

CP = getStat(4,num_tasks,lam,mst,num_trials)

plt.figure(figsize=(15,8))
plt.title("CPU Utilization over "+str(num_trials)+" trials.")
plt.hist(CP,bins=np.linspace(0,1,101),width=0.01,edgecolor='k')
print()


num_tasks_16B = 100
lam_16B = 1.0
mst_16B = 0.7

CP_16B = getStat(4,num_tasks_16B,lam_16B,mst_16B,num_trials)

std_16B = std(CP_16B, ddof=1)
mean_16B = mean(CP_16B)
k_16B = norm.interval(alpha=0.95, loc=0, scale=1)[1]
Sx_16B = std_16B / 10

print("The 95% confidence interval with the mean " + str(np.around(mean_16B,4)) + " is [" + str(np.around(mean_16B-k_16B*Sx_16B,4)) + ", " + str(np.around(mean_16B+k_16B*Sx_16B,4)) + "]")




# Analyzing the Mean Wait Time

num_tasks_17A = 100
lam_17A = 1.0
mst_17A = 0.7

WT = getStat(3,num_tasks_17A,lam_17A,mst_17A,num_trials)

plt.figure(figsize=(15,8))
plt.title("Mean Wait Time over "+str(num_trials)+" trials.")
plt.hist(WT,bins=np.linspace(0,1,101),width=0.01,edgecolor='k')
print()


num_tasks_17B = 100
lam_17B = 1.0
mst_17B = 0.7

WT_17B = getStat(3,num_tasks_17B,lam,mst_17B,100)

std_17B = std(WT_17B,ddof=1)
mean_17B = mean(WT_17B)
k_17B = norm.interval(alpha=0.95, loc=0, scale=1)[1]
Sx_17B = std_17B / 10

print("The 95% confidence interval with the mean " + str(np.around(mean_17B,4)) + " is [" + str(np.around(mean_17B-k_17B*Sx_17B,4)) + ", " + str(np.around(mean_17B+k_17B*Sx_17B,4)) + "]")




# Now we are going to run multiple trials of our simulation with different values of lam (leaving mst = 1.0 for all runs),
# and graph and analyze the results.
# The main issue we want to understand is what happens to various parameters as the arrival rate lam gets
# larger and the mean service time mst exceeds the mean interarrival time beta,
# forcing the system into overload and longer and longer queues.
#
# In concrete terms, arriving tasks will have a mean service time of 1 second,
# and we will investigate mean arrival rates from 0.25 (1 arrival every 4 seconds) to 2.0 (2 arrivals per second).
# The overload point is at lam = 1.0, since after that, there is no way for the system to keep up
# with the workload as a whole.

# Plot the statistic at results[numStat] against the sequence of lam values in lam_list

seed(0)

num_trials_inside_loop = 30           # try this at 20 or 30 if possible

def plotStat(numStat,numTasks,mst,lam_list,titl):

    # for each lam in lams, run the simulation and collect the statistic results[numStat]
    # in a list, then plot these against lams to see the effect of arrival rate
    # on this statistic.

    meanList = []

    for lam in lam_list:
        total = 0
        for i in range(num_trials_inside_loop):
            total += analyzeResults(runSimulation(getTaskList(numTasks,lam,mst)))[numStat]
        meanList.append(total/num_trials_inside_loop)

    plt.figure(figsize=(12, 3))
    plt.title(titl)
    plt.plot(lam_list,meanList)
    plt.show()

numTasks = 500     # Try this at higher numbers, say 300 or 500

lam_list = list(np.arange(0.25,2.0,0.05))   # the different values of lam to use in this problem
mst = 1.0                         # mean service time of arriving tasks -- do not change this parameter


plotStat(4,numTasks,mst,lam_list,titl="Mean CPU Utilization vs Mean Arrival Rate $\lambda$")


# Now plot these parameters vs the mean arrival rate lam:
#
# Mean Wait Time
# Mean Queue Length
# Std of Queue Length


plotStat(3,numTasks,mst,lam_list,titl="Mean Wait Time vs Mean Arrival Rate $\lambda$")
plotStat(5,numTasks,mst,lam_list,titl="Mean Queue Length vs Mean Arrival Rate $\lambda$")
plotStat(6,numTasks,mst,lam_list,titl="Std of Queue Length vs Mean Arrival Rate $\lambda$")
