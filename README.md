# Discrete Event Simulator

Assignment and final lab for "**Distributed Computing**" course.

**Objective**: simulate and analyze a distributed system

> **Note**: all parameters are optional (having default values) unless stated otherwise.

## SIR

Example for understanding system state and events

usage: `python3 ./sir.py [--population] [--infected] [--seed] [--avg-contact-time] [--avg-recovery-time] [--verbose] [--plot_interval]`

- `--population`: *`int`* the total population of individuals to simulate
- `--infected`: *`int`* starting number of infected people
- `--seed`: *`int`* random number generator seed
- `--avg-contact-time`: *`float`* average contact time between two individuals
- `--avg-recovery-time`: *`float`* average recovery time after infection
- `--verbose`: enable verbose mode
- `--plot_interval`: *`float`* data point collection frequency for the final plot

## M/M/n queue

Simulation of a multi server FIFO queueign model.

- Jobs are spawned at a `lambd` rate and are competed with a `mu` probability.
- The simulation supports `n` multiple servers.
- Each spawned job is inserted using a so called "SuperMarket model" in the emptyest queue selected from a `d` subset of `n`
  - The SuperMarket model works just like when checking out at costco; you scan the nearest ~5 queues and select the one with less people waiting.  In the same fashion, the supermarket model watches `d` severs (picked at random) and inserts the job in the queue with less jobs awaiting completion.

usage: `python3 ./mmn_queue.py [--lambd] [--mu] [--max-t] [--n] [--d] [--csv]`

- `--lambd`: *`float`* arrival rate of jobs
- `--mu`: *`float`* service rate of jobs
- `--max-t`: *`float`* maximum simulation time
- `--n`: *`int`* number of simulated servers
- `--d`: *`int`* number of servers to watch for the supermarket model
- `--csv`: *`str`* path to csv file to save the simulation data

## Erasure Coding

*To be implemented :(*
