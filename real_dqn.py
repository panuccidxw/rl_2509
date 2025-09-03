'''
The instability of training in carpole_naive_dqn.py is correlation in two forms:
(P1) between observations
(P2) between policy and observations (chasing one's own tail)
most reinforcement learning problem observations are time-series by nature.
correlation is inherent in time-series problem.
while correlation is conducive for convergence in learning, it carries valuable causal information.
so we have to deal with correlation in two manners:
(A) we keep the good part (causal information) by stacking sequential observation
(B) we eliminate the bad part (correlation) by
(B1) experience replay to address (P1)
(B2) target network to address (P2)
'''

