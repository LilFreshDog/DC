#!/usr/bin/env python

import argparse
import csv
import collections
from random import expovariate, sample

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
        self.schedule(expovariate(lambd), Arrival(0, 0))
    
    def get_min_queue(self) -> int:
        """ returns id of the shortest length queue selected from a subset `d` (supermarket model) """
        sampled_queues_ids = sample(range(self.n), self.d)
        min_queue_id = sampled_queues_ids[0]
        for i in sampled_queues_ids:
            # if self.queue_len(i) < self.queue_len(min_queue_id): # using queue_len()
            if len(self.queues[i]) < len(self.queues[min_queue_id]):
                min_queue_id = i
        return min_queue_id
    
    def print_queue(self):
        """ print number of jobs in each queue """
        print([self.queue_len(i) for i in range(self.n)])

    def schedule_arrival(self, job_id, queue_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id, queue_id))

    def schedule_completion(self, job_id, queue_id):
        # schedule the time of the completion event
        self.schedule(expovariate(self.completion_rate), Completion(job_id, queue_id))

    def queue_len(self, queue_id):
        return (self.running[queue_id] is None) + len(self.queues[queue_id])


class Arrival(Event):

    def __init__(self, job_id: int, queue_id: int):
        self.id = job_id
        self.queue_id = queue_id

    def process(self, sim: MMN):
        #set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        
        # queue_id = sample(range(len(sim.queues)), 1)[0] # uncomment to select a random queue
        queue_id = sim.get_min_queue() # uncomment to select the shortest queue (supermarket model)

        # if there is no running job in the current queue, start the job
        if sim.running[self.queue_id] is None:
            sim.running[self.queue_id] = self.id
            sim.schedule_completion(self.id, self.queue_id)
        # otherwise put the job into the queue
        else:
            #choose the queue with the shortest length randomly
            sim.queues[queue_id].append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1, queue_id)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.7)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000_000)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--d', type=int, default=1) #number of queues to be use as a subset
    parser.add_argument('--csv', help="CSV file in which to store results")
    args = parser.parse_args()

    sim = MMN(args.lambd, args.mu, args.n, args.d)
    sim.run(args.max_t)

    completions = sim.completions
    W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    print(f"Average time spent in the system: {W}")
    print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd) if args.lambd < 1 else 'inf'}")

    if args.csv is not None:
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == '__main__':
    main()
