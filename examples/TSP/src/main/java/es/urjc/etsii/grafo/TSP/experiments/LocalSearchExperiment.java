package es.urjc.etsii.grafo.TSP.experiments;

import es.urjc.etsii.grafo.TSP.algorithms.constructives.TSPRandomConstructive;
import es.urjc.etsii.grafo.TSP.algorithms.neighborhood.SwapNeighborhood;
import es.urjc.etsii.grafo.TSP.model.TSPInstance;
import es.urjc.etsii.grafo.TSP.model.TSPSolution;
import es.urjc.etsii.grafo.solver.SolverConfig;
import es.urjc.etsii.grafo.solver.algorithms.Algorithm;
import es.urjc.etsii.grafo.solver.algorithms.SimpleAlgorithm;
import es.urjc.etsii.grafo.solver.improve.ls.LocalSearchFirstImprovement;
import es.urjc.etsii.grafo.solver.services.AbstractExperiment;

import java.util.ArrayList;
import java.util.List;

public class LocalSearchExperiment extends AbstractExperiment<TSPSolution, TSPInstance> {

    public LocalSearchExperiment(SolverConfig solverConfig) {
        super(solverConfig.isMaximizing());
    }

    @Override
    public List<Algorithm<TSPSolution, TSPInstance>> getAlgorithms() {

        var algorithms = new ArrayList<Algorithm<TSPSolution, TSPInstance>>();


        algorithms.add(new SimpleAlgorithm<>(new TSPRandomConstructive(), new LocalSearchFirstImprovement<>(super.isMaximizing(), new SwapNeighborhood())));



        return algorithms;
    }
}
