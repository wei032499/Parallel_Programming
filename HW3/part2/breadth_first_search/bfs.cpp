#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include <vector>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{

// clang-format off
    #pragma omp parallel
    {
        std::vector<int> vertices;
        vertices.reserve(new_frontier->max_vertices);
        #pragma omp for
        for (int i = 0; i < frontier->count; i++)
        {

            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];
            
            
    
            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] != NOT_VISITED_MARKER)
                    continue;
                
                __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1);
                vertices.push_back(outgoing);
            }


        }

        int index = __sync_fetch_and_add(&new_frontier->count, vertices.size());
        std::copy(vertices.begin(),vertices.end(),new_frontier->vertices + index);
        
    }

}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    bool *visited,
    int *distances)
{
    #pragma omp parallel
    {
        std::vector<int> vertices;
        vertices.reserve(frontier->max_vertices);
        // bool finished = true;
        // int *vertices = new int[frontier->max_vertices];
        // int count = 0;
    
        #pragma omp for
        for (int i = 0; i < g->num_nodes; i++)
        {
            // if(distances[i] != NOT_VISITED_MARKER)
            if(visited[i])
                continue;
            const Vertex* in_begin = incoming_begin(g, i);
            const Vertex* out_begin = incoming_end(g, i);
            for (const Vertex* neighbor=in_begin; neighbor!=out_begin; neighbor++)
            {
                // if (distances[*neighbor] == current_distance)
                if(!visited[*neighbor])
                    continue;
                    
                distances[i] = distances[*neighbor] + 1;
                vertices.push_back(i);
                // finished = false;
                // vertices[count++] = i;
                break;
                
            }
            
        }

        int index = __sync_fetch_and_add(&frontier->count, vertices.size());
        std::copy(vertices.begin(),vertices.end(),frontier->vertices + index);
        // int index = __sync_fetch_and_add(&frontier->count, count);
        // memcpy(frontier->vertices + index, vertices, sizeof(int)*count);
        // delete[] vertices;

        // for (int i = 0; i < vertices.size(); i++)
        //     visited[vertices[i]] = true;
        // if(!finished)
        //     frontier->count = 1;

        
        
    }

}

void bfs_bottom_up(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    bool *visited = new bool[graph->num_nodes]{false};

    visited[ROOT_NODE_ID] = true;


    while(frontier->count != 0)
    {
        vertex_set_clear(frontier);
        bottom_up_step(graph, frontier, visited, sol->distances);

        // #pragma omp parallel for
        for(int i=0;i<frontier->count;i++)
            visited[frontier->vertices[i]] = true;

    }

    delete[] visited;

    

}

void bfs_hybrid(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // clang-format on

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    bool *visited = new bool[graph->num_nodes]{false};
    visited[ROOT_NODE_ID] = true;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        if (frontier->count < graph->num_nodes * 0.1)
            top_down_step(graph, frontier, new_frontier, sol->distances);
        else
            bottom_up_step(graph, new_frontier, visited, sol->distances);

        // clang-format off
        // #pragma omp parallel for
        for (int i = 0; i < new_frontier->count; i++)
            visited[new_frontier->vertices[i]] = true;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    delete[] visited;
}
