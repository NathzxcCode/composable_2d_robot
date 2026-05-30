"""
GTSAM Copyright 2010-2018, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Simple robot motion example, with prior and two odometry measurements
Author: Frank Dellaert
"""
# pylint: disable=invalid-name, E1101

from __future__ import print_function

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
import torch

from gbp_utilities import Gaussian

class Node:
    def __init__(self, dim, eta, lam, adj_factors, ID):
        self.dofs = dim
        self.belief = Gaussian(dim, eta, lam)
        self.adj_factors = adj_factors
        self.ID = ID

    def update_belief(self) -> None:
        """ Update local belief estimate by taking product of all incoming messages along all edges. """
        self.belief.eta = torch.zeros(self.dofs, dtype=torch.float64)
        self.belief.lam = torch.eye(self.dofs, dtype=torch.float64) * 1e-6
        
        for factor_id in self.adj_factors:  # messages from other adjacent variables
            factor = factors[factor_id]
            message_ix = factor.adj_vIDs.index(self.ID)
            self.belief.eta += factor.messages[message_ix].eta
            self.belief.lam += factor.messages[message_ix].lam

        # print(self.belief.eta)

    def get_delta(self):
        # Solve for the mean update: delta_x = inv(Lambda) * eta
        delta_x = torch.linalg.solve(self.belief.lam, self.belief.eta)
        return delta_x

class Factor:
    def __init__(self, node_dofs, adj_vIDs):
        self.node_dofs = node_dofs # ordered list of connected nodes and their dofs eg: [3,3] for PriorFactorPose2
        self.adj_vIDs = adj_vIDs
        self.messages = [Gaussian(dof) for dof in node_dofs]

    def reset_messages(self):
        self.messages = [Gaussian(dof) for dof in self.node_dofs]

    def compute_messages(self, eta, lam, damping: float = 0.) -> None:
        """ Compute all outgoing messages from the factor. """
        messages_eta, messages_lam = [], []

        start_dim = 0
        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = eta.clone().double(), lam.clone().double()

            # Take product of factor with incoming messages
            start = 0
            for var, adj_node_id in enumerate(self.adj_vIDs):
                if var != v:
                    var_dofs = nodes[adj_node_id].dofs
                    eta_factor[start:start + var_dofs] += nodes[adj_node_id].belief.eta - self.messages[var].eta
                    lam_factor[start:start + var_dofs, start:start + var_dofs] += nodes[adj_node_id].belief.lam - self.messages[var].lam
                start += nodes[adj_node_id].dofs

            # Divide up parameters of distribution
            mess_dofs = nodes[self.adj_vIDs[v]].dofs
            eo = eta_factor[start_dim:start_dim + mess_dofs]
            eno = torch.cat((eta_factor[:start_dim], eta_factor[start_dim + mess_dofs:]))

            loo = lam_factor[start_dim:start_dim + mess_dofs, start_dim:start_dim + mess_dofs]
            lono = torch.cat((lam_factor[start_dim:start_dim + mess_dofs, :start_dim],
                              lam_factor[start_dim:start_dim + mess_dofs, start_dim + mess_dofs:]), dim=1)
            lnoo = torch.cat((lam_factor[:start_dim, start_dim:start_dim + mess_dofs],
                              lam_factor[start_dim + mess_dofs:, start_dim:start_dim + mess_dofs]), dim=0)
            lnono = torch.cat(
                        (
                            torch.cat((lam_factor[:start_dim, :start_dim], lam_factor[:start_dim, start_dim + mess_dofs:]), dim=1),
                            torch.cat((lam_factor[start_dim + mess_dofs:, :start_dim], lam_factor[start_dim + mess_dofs:, start_dim + mess_dofs:]), dim=1)
                        ),
                        dim=0
                    )

            new_message_lam = loo - lono @ torch.inverse(lnono) @ lnoo
            new_message_eta = eo - lono @ torch.inverse(lnono) @ eno
            messages_eta.append((1 - damping) * new_message_eta + damping * self.messages[v].eta)
            messages_lam.append((1 - damping) * new_message_lam + damping * self.messages[v].lam)
            start_dim += nodes[self.adj_vIDs[v]].dofs

        for v in range(len(self.adj_vIDs)):
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]

nodes = {} # node_id : Node
factors = [] # ordered by factor_id, type Factor

# Create noise models
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))


def main():
    """Main runner"""
    # Create an empty nonlinear factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Add a prior on the first pose, setting it to the origin
    # A prior factor consists of a mean and a noise model (covariance matrix)
    priorMean = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
    graph.add(gtsam.PriorFactorPose2(1, priorMean, PRIOR_NOISE))
    factors.append(Factor(node_dofs=[3], adj_vIDs=[1]))

    # Add odometry factors
    odometry = gtsam.Pose2(2.0, 0.0, 0.0)
    # For simplicity, we will use the same noise model for each odometry factor
    # Create odometry (Between) factors between consecutive poses
    graph.add(gtsam.BetweenFactorPose2(1, 2, odometry, ODOMETRY_NOISE))
    factors.append(Factor(node_dofs=[3,3], adj_vIDs=[1,2]))
    graph.add(gtsam.BetweenFactorPose2(2, 3, odometry, ODOMETRY_NOISE))
    factors.append(Factor(node_dofs=[3,3], adj_vIDs=[2,3]))
    print("\nFactor Graph:\n{}".format(graph))

    # Create the data structure to hold the initialEstimate estimate to the solution
    # For illustrative purposes, these have been deliberately set to incorrect values
    initial = gtsam.Values()
    initial.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    nodes[1] = Node(dim=3, eta=torch.zeros(3, dtype=torch.float64), lam=torch.eye(3, dtype=torch.float64) * 1e-6, adj_factors=[0,1], ID=1)
    initial.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    nodes[2] = Node(dim=3, eta=torch.zeros(3, dtype=torch.float64), lam=torch.eye(3, dtype=torch.float64) * 1e-6, adj_factors=[1,2], ID=2)
    initial.insert(3, gtsam.Pose2(4.1, 0.1, 0.1))
    nodes[3] = Node(dim=3, eta=torch.zeros(3, dtype=torch.float64), lam=torch.eye(3, dtype=torch.float64) * 1e-6, adj_factors=[2], ID=3)
    print("\nInitial Estimate:\n{}".format(initial))

    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print(result)

    N_OUTER = 10        # relinearization steps
    N_INNER = 10        # GBP message-passing iterations per linearization
    for outer in range(N_OUTER):
        # Step 1: Linearize at current estimate
        gaussian_graph = graph.linearize(initial)
        # Step 2: Reset all messages (new linearization = fresh start)
        for factor in factors:
            factor.reset_messages()
        # Reset node beliefs too
        for node_obj in nodes.values():
            node_obj.belief.eta = torch.zeros(node_obj.dofs, dtype=torch.float64)
            node_obj.belief.lam = torch.eye(node_obj.dofs, dtype=torch.float64) * 1e-6
        # Step 3: Extract eta, lam for each factor once
        factor_eta_lam = []
        for factor_id in range(gaussian_graph.size()):
            factor = gaussian_graph.at(factor_id)
            A, b = factor.jacobian()
            lam = torch.from_numpy(factor.information())
            eta = torch.from_numpy(A.T @ b).flatten()
            factor_eta_lam.append((eta, lam))
        # Step 4: Run GBP inner loop on the FIXED linearized graph
        for inner in range(N_INNER):
            for factor_id, factor_obj in enumerate(factors):
                eta, lam = factor_eta_lam[factor_id]
                factor_obj.compute_messages(eta, lam)
            for node_obj in nodes.values():
                node_obj.update_belief()
        # Step 5: Retract once using the converged GBP beliefs
        deltas = gtsam.VectorValues()
        for node_id, node_obj in nodes.items():
            delta_x = node_obj.get_delta()
            deltas.insert(node_id, delta_x)
        initial = initial.retract(deltas)

    print(initial)

    # print(gaussian_graph)

    # print("\nFinal Result:\n{}".format(result))

    # # 5. Calculate and print marginal covariances for all variables
    # marginals = gtsam.Marginals(graph, result)
    # for i in range(1, 4):
    #     print("X{} covariance:\n{}\n".format(i,
    #                                          marginals.marginalCovariance(i)))

    # for i in range(1, 4):
    #     gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5,
    #                           marginals.marginalCovariance(i))
    # plt.axis('equal')
    # plt.show()


if __name__ == "__main__":
    main()