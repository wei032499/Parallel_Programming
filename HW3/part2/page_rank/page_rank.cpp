#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  int numNodes = num_nodes(g);
  double *score_old = new double[numNodes];
  double *confer = new double[numNodes];
  double global_diff;
  bool converged = false;

  double x; // no outgoing edges

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  double equal_prob = 1.0 / numNodes;

  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  while (!converged)
  {

    memcpy(score_old, solution, sizeof(double) * numNodes);

    x = 0;

    // clang-format off
    #pragma omp parallel for reduction(+:x)
    for (int i = 0; i < numNodes; ++i)
    {
      confer[i] = score_old[i] / outgoing_size(g, i);
      
      if (outgoing_size(g, i) == 0)
        x += damping * score_old[i] / numNodes;
    }

    global_diff = 0;

    #pragma omp parallel for reduction(+:global_diff)
    for (int i = 0; i < numNodes; ++i)
    {
      const Vertex *in_begin = incoming_begin(g, i);
      const Vertex *in_end = incoming_end(g, i);

      solution[i] = 0;
      
      for (const Vertex *v = in_begin; v != in_end; v++)
      {
        solution[i] += confer[*v];
      }
      solution[i] = (damping * solution[i]) + (1.0 - damping) / numNodes;

      solution[i] += x;

      global_diff += abs(solution[i] - score_old[i]);
    }

    converged = (global_diff < convergence);

  }

  delete[] score_old;
  delete[] confer;
}
