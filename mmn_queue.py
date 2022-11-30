#!/usr/bin/env python

import argparse
import csv
import collections
from random import expovariate, sample, randint

from discrete_event_sim import Simulation, Event

# To use weibull variates, for a given set of parameter do something like
# from weibull import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


class MMN(Simulation):

    def __init__(self, lambd, mu, n, d):
        super().__init__()
        self.running = [None] * n 
        #self.queue = collections.deque()  # FIFO queue of the system
        self.queues = [collections.deque() for i in range(n)]  # FIFO queues of the system
        self.queue_counter = [0] * n # average length of the queues
        self.total_jobs = 0  # total number of jobs ever entered in the system (?)
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd # arrival rate
        self.mu = mu # completion rate
        self.n = n if n > 0 else 1 # number of queues
        if 0 < d <= n:
            self.d = d
        else:
            print("\nINFO:\n  'd' must be in the range (0, n]\n  automatically set 'd' to 'n'\n")
            self.d = n
        self.arrival_rate = lambd * n
        self.completion_rate = mu 
        self.schedule(expovariate(lambd), Arrival(0))
    
    def get_min_queue(self) -> int:
        """ returns id of the shortest length queue selected from a subset `d` (supermarket model) """
        sampled_queues_ids = sample(range(self.n), self.d)
        min_queue_id = 0
        for i in sampled_queues_ids:
            if len(self.queues[i]) < len(self.queues[min_queue_id]):
                min_queue_id = i
        return min_queue_id
    
    def print_queue(self):
        """ print number of jobs in each queue """
        print([self.queue_len(i) for i in range(self.n)], end="\n")

    def schedule_arrival(self, job_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self, job_id, queue_id):
        # schedule the time of the completion event
        self.schedule(expovariate(self.completion_rate), Completion(job_id, queue_id))

    def queue_len(self, queue_id):
        return (self.running[queue_id] is None) + len(self.queues[queue_id])


class Arrival(Event):

    def __init__(self, job_id: int):
        self.id = job_id

    def process(self, sim: MMN):
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        sim.total_jobs += 1
        
        # queue_id = randint(0, sim.n) # uncomment to select a random queue
        queue_id = sim.get_min_queue() # uncomment to select the shortest queue (supermarket model)
        sim.queue_counter[queue_id] += 1

        # if there is no running job in the current queue, start the job
        if sim.running[queue_id] is None:
            sim.running[queue_id] = self.id
            sim.schedule_completion(self.id, queue_id)
        # otherwise put the job into the queue
        else:
            #choose the queue with the shortest length randomly
            sim.queues[queue_id].append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)

class Completion(Event):
    def __init__(self, job_id, queue_id):
        self.id = job_id  # currently unused, might be useful when extending
        self.queue_id = queue_id

    def process(self, sim: MMN):
        assert sim.running[self.queue_id] is not None
        # DEBUG 
        sim.print_queue() if sim.n <= 10 else None
        # set the completion time of the running job
        sim.completions[sim.running[self.queue_id]] = sim.t
        # if the queue is not empty
        if len(sim.queues[self.queue_id]) > 0:
            # get a job from the queue
            sim.running[self.queue_id] = sim.queues[self.queue_id].popleft()
            # schedule its completion
            sim.schedule_completion(sim.running[self.queue_id], self.queue_id)
        else:
            sim.running[self.queue_id] = None


def show_graphs(result_queues: list, average_queue_lengths: list, lambdas: list, max_t: int):
    import matplotlib.pyplot as plt
    import numpy as np
    import collections

    print("Creating graphs...")
    counters = []

    for i, queue in enumerate(average_queue_lengths):
        #queue_lengths = [len(queue) for queue in result_queues[i] if len(queue) > 0]  
        av_queue_lengths = [length/max_t for length in average_queue_lengths[i] if len(queue) > 0]
        counter = collections.Counter(av_queue_lengths)
        counter = collections.OrderedDict(sorted(counter.items()))
        for key,value in counter.items():
            counter[key] = value/len(av_queue_lengths)
        counters.append(counter)

    plt.title("Distribution of queue lengths")
    plt.plot(*zip(*sorted(counters[0].items())), label="lambda = {}".format(lambdas[0]))
    plt.plot(*zip(*sorted(counters[1].items())), label="lambda = {}".format(lambdas[1]))
    plt.plot(*zip(*sorted(counters[2].items())), label="lambda = {}".format(lambdas[2]))
    plt.plot(*zip(*sorted(counters[3].items())), label="lambda = {}".format(lambdas[3]))
    plt.xlim(0, 14)
    plt.ylim(0, 1)
    plt.xlabel("Queue length")
    plt.ylabel("Fraction of queues with at leas that size")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.7)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000_000)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--d', type=int, default=1) #number of queues to be use as a subset
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument('--graph', action='store_true', default=False, help="Enable graphing of results")
    args = parser.parse_args()

    result_queues = []
    average_queues_lenghts = []
    lambdas_x = [0.5, 0.90,  0.95, 0.99]
    for lambd_x in lambdas_x:
        sim = MMN(lambd_x, args.mu, args.n, args.d)
        sim.run(args.max_t)

        completions = sim.completions
        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        print(f"Average time spent in the system: {W}")
        print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd ) if args.lambd < 1 else 'inf'}")

        if args.csv is not None:
            with open(args.csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([args.lambd, args.mu, args.max_t, W])

        result_queues.append(sim.queues)
        average_queues_lenghts.append(sim.queue_counter)
        print(sim.queue_counter)
        print([counter/sim.total_jobs for counter in sim.queue_counter], end=" ")
        print()
    if args.graph:
        show_graphs(result_queues, average_queues_lenghts, lambdas_x, sim.total_jobs)



if __name__ == '__main__':
    main()
