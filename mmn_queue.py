#!/usr/bin/env python

import argparse
import csv
import collections
from random import expovariate, sample

from discrete_event_sim import Simulation, Event

class MMN(Simulation):

    def __init__(self, lambd, mu, n, d):
        super().__init__()
        self.running = [None] * n 
        self.queues = [collections.deque() for i in range(n)] # FIFO queues of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.queue_lengths = {}
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

    def update_queue_lenghts_dict(self, sampled_queues_ids: list):
        for i in sampled_queues_ids:
            #if len(self.queues[i]) != 0:
            if len(self.queues[i]) in self.queue_lengths:
                self.queue_lengths[len(self.queues[i])] += 1
            else:
                self.queue_lengths[len(self.queues[i])] = 1
    
    def get_min_queue(self) -> int:
        """ returns id of the shortest length queue selected from a subset `d` (supermarket model) """
        sampled_queues_ids = sample(range(self.n), self.d)

        # updating queue lenghts dict for graph
        self.update_queue_lenghts_dict(sampled_queues_ids)

        min_queue_id = sampled_queues_ids[0]
        for i in sampled_queues_ids:
            if len(self.queues[i]) < len(self.queues[min_queue_id]):
                min_queue_id = i
        return min_queue_id
    
    def print_queues(self):
        """ print number of jobs in each queue """
        print([self.queue_len(i) for i in range(self.n)], end="\r")

    def schedule_arrival(self, job_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self, job_id, queue_id):
        # schedule the time of the completion event
        self.schedule(expovariate(self.completion_rate), Completion(job_id, queue_id))

    def queue_len(self, queue_id):
        return len(self.queues[queue_id])


class Arrival(Event):

    def __init__(self, job_id: int):
        self.id = job_id

    def process(self, sim: MMN):
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        
        # queue_id = sample(range(len(sim.queues)), 1)[0] # uncomment to select a random queue
        queue_id = sim.get_min_queue() # uncomment to select the shortest queue (supermarket model)

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
        # DEBUG # sim.print_queue() if sim.n <= 10 else None
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


def show_single_graph(sim: MMN):
    import matplotlib.pyplot as plt
    
    sum_all_queue_lenghts = sum(sim.queue_lengths.values())
    normalized_queues_lengths = {}
    acc = 0
    for length, occurrences in sorted(sim.queue_lengths.items()):
        normalized_queues_lengths[length] = (sum_all_queue_lenghts - acc) / sum_all_queue_lenghts
        acc += occurrences

    plt.plot(list(normalized_queues_lengths.keys()), list(normalized_queues_lengths.values()))
    plt.grid(True)
    plt.xlim(0, 14)
    # plt.ylim(0, 1)
    plt.xlabel("Queue length")
    plt.ylabel("Fraction of queues with at leas that size")
    plt.title(str(sim.d) + " choice" + ("s" if sim.d != 1 else ""))
    plt.show()


def show_all_graphs(lambdas, simulation_queue_lengths, d):
    import matplotlib.pyplot as plt

    for i in range(len(lambdas)):
        final_results= dict(sorted(simulation_queue_lengths[i].items(), key=lambda item: item[0]))
        final_results = {k: v/sum(final_results.values()) for k, v in final_results.items()}
        
        sum_all_queue_lenghts = sum(simulation_queue_lengths[i].values())
        normalized_queues_lengths = {}
        acc = 0
        for length, occurrences in sorted(simulation_queue_lengths[i].items()):
            normalized_queues_lengths[length] = (sum_all_queue_lenghts - acc) / sum_all_queue_lenghts
            acc += occurrences
        
        plt.plot(list(normalized_queues_lengths.keys()), list(normalized_queues_lengths.values()), label="lambda = " + str(lambdas[i]))
    
    plt.grid(True)
    plt.xlim(0, 14)
    plt.ylim(0, 1)
    plt.xlabel("Queue length")
    plt.ylabel("Fraction of queues with at leas that size")
    plt.title("")
    plt.title(str(d) + " choice" + ("s" if d != 1 else ""))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, help="jobs arrival rate")
    parser.add_argument('--mu', type=float, default=1, help="jobs completion rate")
    parser.add_argument('--max-t', type=float, default=1_000_000, help="maximum simulation time")
    parser.add_argument('--n', type=int, default=1, help="number of simulated queues")
    parser.add_argument('--d', type=int, default=1, help="number of queues to monitor when choosing the min queue") #number of queues to be use as a subset
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument('--graph', action='store_true', default=False, help="Enable graphing of results")
    parser.add_argument('--run-all', action='store_true', default=False, help="Run simulations for all lambdas (0.5, 0.9, 0.95, 0.99)")
    args = parser.parse_args()

    if args.lambd is None:
        args.lambd = 0.5

    if args.run_all:
        lambdas = [0.5, 0.9, 0.95, 0.99]
    else:
        lambdas = [args.lambd]
    
    simulation_queue_lengths = []
    for lambd in lambdas:
        sim = MMN(lambd, args.mu, args.n, args.d)
        sim.run(args.max_t)

        completions = sim.completions
        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        print(f"Average time spent in the system: {W}")
        print(f"Theoretical expectation for random server choice: {1 / (1 - lambd) if lambd < 1 else 'inf'}")

        if args.csv is not None:
            with open(args.csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([lambd, args.mu, args.max_t, W])
        
        simulation_queue_lengths.append(sim.queue_lengths)

    if args.run_all:
        show_all_graphs(lambdas, simulation_queue_lengths, args.d)
    elif args.graph:
        show_single_graph(sim)


if __name__ == '__main__':
    main()
