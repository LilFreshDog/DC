#!/usr/bin/env python

import argparse
import csv
import collections
from queue import Empty
from random import expovariate

from discrete_event_sim import Simulation, Event

# To use weibull variates, for a given set of parameter do something like
# from weibull import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


class MMN(Simulation):

    def __init__(self, lambd, mu, n):
        
        super().__init__()
        self.running = None  # if not None, the id of the running job
        #self.queue = collections.deque()  # FIFO queue of the system
        self.queues = [collections.deque() for _ in range(n)] # new multiple queues
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
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
        sim.arrivals[self.id] = sim.t + sim.arrival_rate
        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running is None:
            sim.running = self.id
            sim.schedule_completion(self.id)
        # otherwise put the job into the queue
        else:
            #min_queue = sim.queues[0] #[queue for queue in sim.queues if len(queue) == min([len(queue) for queue in sim.queues])]
            min_queue = None
            min_length = 100_000_000_000_000
            for queue in sim.queues:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.7)         # prob. of new job arrival
    parser.add_argument('--mu', type=float, default=1)              # prob. of job completion
    parser.add_argument('--max-t', type=float, default=1_000_000)   # max. simulated time
    parser.add_argument('--n', type=int, default=1)                 # number of simulated servers
    parser.add_argument('--csv', help="CSV file in which to store results")
    args = parser.parse_args()

    sim = MMN(args.lambd, args.mu, args.n)
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
