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
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.arrival_rate = lambd * n
        self.completion_rate = mu 
        self.schedule(expovariate(lambd), Arrival(0, 0))

    def schedule_arrival(self, job_id, queue_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule( expovariate(self.arrival_rate),Arrival(job_id, queue_id))

    def schedule_completion(self, job_id, queue_id):
        # schedule the time of the completion event
        self.schedule(expovariate(self.completion_rate), Completion(job_id, queue_id))

    @property
    def queue_len(self):
        return (self.running is None) + len(self.queue)


class Arrival(Event):

    def __init__(self, job_id: int, queue_id: int):
        self.id = job_id
        self.queue_id = queue_id

    def process(self, sim: MMN):
        #set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        random_queue_id = sample(range(len(sim.queues)), 1)[0]

        # if there is no running job in the current queue, start the job
        if sim.running[self.queue_id] is None:
            sim.running[self.queue_id] = self.id
            sim.schedule_completion(self.id, self.queue_id)
        # otherwise put the job into the queue
        else:
            #choose the queue with the shortest length randomly
            sim.queues[random_queue_id].append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1, random_queue_id)

class Completion(Event):
    def __init__(self, job_id, queue_id):
        self.id = job_id  # currently unused, might be useful when extending
        self.queue_id = queue_id

    def process(self, sim: MMN):
        assert sim.running[self.queue_id] is not None
        # set the completion time of the running job
        sim.completions[sim.running[self.queue_id]] = sim.t + sim.mu
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
    print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd)}")

    if args.csv is not None:
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == '__main__':
    main()
