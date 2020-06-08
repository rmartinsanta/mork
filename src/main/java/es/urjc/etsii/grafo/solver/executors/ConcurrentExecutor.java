package es.urjc.etsii.grafo.solver.executors;

import es.urjc.etsii.grafo.io.Instance;
import es.urjc.etsii.grafo.io.Result;
import es.urjc.etsii.grafo.io.WorkingOnResult;
import es.urjc.etsii.grafo.solution.Solution;
import es.urjc.etsii.grafo.solver.algorithms.Algorithm;
import es.urjc.etsii.grafo.solver.create.SolutionBuilder;
import es.urjc.etsii.grafo.solver.services.ExceptionHandler;
import es.urjc.etsii.grafo.util.ConcurrencyUtil;
import es.urjc.etsii.grafo.util.RandomManager;
import org.springframework.beans.factory.annotation.Value;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class ConcurrentExecutor<S extends Solution<I>, I extends Instance> extends Executor<S,I>{

    private static final Logger logger = Logger.getLogger(ConcurrentExecutor.class.getName());

    private final int nWorkers;
    private final ExecutorService executor;

    public ConcurrentExecutor(@Value("${solver.nWorkers:-1}") int nWorkers) {
        if(nWorkers == -1){
            this.nWorkers = Runtime.getRuntime().availableProcessors() / 2;
        } else {
            this.nWorkers = nWorkers;
        }
        this.executor = Executors.newFixedThreadPool(this.nWorkers);
    }

    @Override
    public Collection<Result> execute(I ins, int repetitions, List<Algorithm<S,I>> list, SolutionBuilder<S,I> solutionBuilder, ExceptionHandler<S,I> exceptionHandler) {

        Map<Algorithm<S, I>, WorkingOnResult> resultsMap = new ConcurrentHashMap<>();

        logger.info("Submitting tasks for instance: "+ins.getName());
        for(var algorithm: list){
            var futures = new ArrayList<Future<Object>>();
            resultsMap.put(algorithm, new WorkingOnResult(repetitions, algorithm.toString(), ins.getName()));
            for (int i = 0; i < repetitions; i++) {
                int _i = i;
                futures.add(executor.submit(() -> {
                    try {
                        RandomManager.reset(_i);
                        long starTime = System.nanoTime();
                        var solution = algorithm.algorithm(ins, solutionBuilder);
                        long endTime = System.nanoTime();
                        long timeToTarget = solution.getLastModifiedTime() - starTime;
                        long ellapsedTime = endTime - starTime;
                        resultsMap.get(algorithm).addSolution(solution, ellapsedTime, timeToTarget);
                        System.out.format("\t%s.\tTime: %.3f (s) \tTTT: %.3f (s) \t%s -- \n", _i+1, ellapsedTime / 1000_000_000D, timeToTarget / 1000_000_000D, solution);

                    } catch (Exception e){
                        exceptionHandler.handleException(e, ins, algorithm);
                    }
                    return null;
                }));
            }
            logger.info(String.format("Waiting for combo instance %s, algorithm %s ",ins.getName(), algorithm.toString()));
            ConcurrencyUtil.awaitAll(futures);
        }

        return resultsMap.values().stream().map(WorkingOnResult::finish).collect(Collectors.toList());
    }

    @Override
    public void shutdown() {
        logger.info("Shutting down executor");
        this.executor.shutdown();
    }
}
