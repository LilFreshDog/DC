#!/usr/bin/env python

import argparse
import csv
import collections
from queue import Empty
from random import expovariate, sample

from discrete_event_sim import Simulation, Event

# To use weibull variates, for a given set of parameter do something like
# from weibull import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


class MMN(Simulation):

    def __init__(self, lambd:float, mu:float, n:int, d:float):
        
        super().__init__()
        self.running = None  # if not None, the id of the running job
        self.queues = [collections.deque() for _ in range(n)] # FIFO queues of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.d = int(n * d) # number of queues to monitor (supermarket model)
        if self.d < 1: self.d = 1
        self.mu = mu
        self.arrival_rate = lambd / n
        self.completion_rate = mu / n
        self.schedule(expovariate(lambd), Arrival(0))

    def schedule_arrival(self, job_id):
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"   
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self, job_id):
        # schedule the time of the completion event
        self.schedule(self.completion_rate, Completion(job_id))

    @property
    def queue_len(self):
        return (self.running is None) + len(self.queues[0])

    @property
    def queue_len(self, n):
        if n < self.n:
            return (self.running is None) + len(self.queues[n])

class Arrival(Event):

    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN):
        #print("process arrival:", self.id)
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
    
        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running is None:
            sim.running = self.id
            sim.schedule_completion(self.id)
        # otherwise put the job into the queue (DONE: check just a subset d of queues)
        else:
            # supermarket model implementation for n queues
            # we are monitoring a subset of queues (d % n)
            sampled_queues = sample(sim.queues, sim.d)
            min_queue = sampled_queues[0]
            min_length = len(sampled_queues[0])
            for queue in sampled_queues:
                if len(queue) < min_length:
                    min_length = len(queue)
                    min_queue = queue
            min_queue.append(self.id)

        # schedule the arrival of the next job
        sim.schedule_arrival(self.id + 1)


class Completion(Event):

    def __init__(self, job_id):
        self.id = job_id  # currently unused, might be useful when extending

    def process(self, sim: MMN):
        #print("process completion:", self.id)
        assert sim.running is not None
        # set the completion time of the running job
        sim.completions[sim.running] = sim.t + sim.completion_rate
        # if the queue is not empty
        if [queue for queue in sim.queues if len(queue) != 0]:
            job_to_complete = None #[queue for queue in sim.queues if queue[0] == min([ queue[0] for queue in sim.queues])] # is not empty
            min = 100_000_000_000_000
            for queue in sim.queues:
                if len(queue) != 0 and queue[0] < min:
                    min = queue[0]
                    job_to_complete = queue
            # get a job from the queues
            sim.running = job_to_complete.popleft()
            # schedule its completion
            sim.schedule_completion(sim.running)
        else:
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
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == '__main__':
    main()
