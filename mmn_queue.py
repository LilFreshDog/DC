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

    def __init__(self, lambd:float=0.7, mu:float=1, n:int=1, d:float=1):
        """MMN constructor

        Args:
            lambd: arrival rate of job in server
            mu: completion rate of job in server
            n: number of servers to simulate
            d: percentage of jobs to be servers to monitor in supermarket model
        """
        
        super().__init__()
        self.running = None # if not None, the id of the running job
        self.queues = [collections.deque() for _ in range(n)] # FIFO queues of the system
        self.arrivals = {} # dictionary mapping job id to arrival time
        self.completions = {} # dictionary mapping job id to completion time
        self.lambd = lambd
        self.mu = mu
        self.n = n if n >=1 else 1
        self.d = int(n * d) if n * d >= 1 else 1 # number of queues to monitor (supermarket model)
        self.arrival_rate = lambd * n
        self.completion_rate = mu
        self.schedule(expovariate(self.arrival_rate), Arrival(0, 0))

    def schedule_arrival(self, job_id, queue_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id, queue_id))

    def schedule_completion(self, job_id, queue_id):
        # schedule the time of the completion event
        self.schedule(self.completion_rate, Completion(job_id, queue_id))

    @property
    def queue_len(self):
        return (self.running is None) + len(self.queues[0])

    @property
    def queue_len(self, n):
        if n < self.n:
            return (self.running is None) + len(self.queues[n])

class Arrival(Event):

    def __init__(self, job_id, queue_id):
        self.id = job_id
        self.queue_id = queue_id

    def process(self, sim: MMN):
        #print("process arrival:", self.id, end='\r')
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        # supermarket model implementation for n queues
        # we are monitoring a subset of queues (d % n)

        sampled_queues_indexes = sample(range(len(sim.queues)), sim.d)    
        min_queue = sim.queues[sampled_queues_indexes[0]]
        min_length = len(min_queue)
        min_index = sampled_queues_indexes[0]

        for queue_index in sampled_queues_indexes :
            if len(sim.queues[queue_index]) < min_length:
                min_length = len(sim.queues[queue_index])
                min_queue = sim.queues[queue_index]
                min_index = queue_index

        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running is None:
            sim.running = self.id
            sim.schedule_completion(self.id, min_index)
        # otherwise put the job into the queue (DONE: check just a subset d of queues)
        else:
            min_queue.append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1, min_index )


class Completion(Event):

    def __init__(self, job_id, queue_id):
        self.id = job_id  # currently unused, might be useful when extending
        self.queue_id = queue_id

    def process(self, sim: MMN):
        #print("process completion:", self.id, end='\r')
        assert sim.running is not None
        # set the completion time of the running job
        sim.completions[sim.running] = sim.t + sim.completion_rate
        # if the queue is not empty
        # check if all queus are empty
        if any(len(queue) > 0 for queue in sim.queues):
            #if not take the job from the queuse with the least id and schedule its completion
            min_queue = None
            min_index = 0

            for queue in sim.queues:
                if len(queue) > 0:
                    if min_queue is None or queue[0] < min_queue[0]:
                        min_queue = queue
                        min_index = sim.queues.index(queue)

            sim.running = min_queue.popleft()
            sim.schedule_completion(sim.running, min_index)

        else:    
            #if yes sim.running = None because there are no jobs to be scheduled
            sim.running = None

def main():
    parser = argparse.ArgumentParser(
        prog = 'mmn_queue',
        description = 'Simulation of a multi-server system using a M/M/n queue',
        epilog = '2022 UniGe - 👊 github.com/GiorgioRen x github.com/thaMilo 👊'
        )
    parser.add_argument('--lambd',  type=float, default=0.7,        help="The probability of new job arrival")
    parser.add_argument('--mu',     type=float, default=1,          help="The probability of job completion")
    parser.add_argument('--max-t',  type=float, default=1_000_000,  help="Max simulated time")
    parser.add_argument('--n',      type=int,   default=1,          help="Number of simulated servers")
    parser.add_argument('--d',      type=float, default=1,          help="Percentage of queues to be monitored in supermarket model")
    parser.add_argument('--csv',                                    help="CSV file in which to store results")
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
            writer.writerow([args.lambd, args.mu, args.n, args.d, args.max_t, W])


if __name__ == '__main__':
    main()
