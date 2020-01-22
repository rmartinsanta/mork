package solver.algorithms;

import io.Instance;
import io.Result;
import solution.ConstructiveNeighborhood;
import solution.Solution;
import solver.create.Constructor;
import solver.create.SolutionBuilder;
import solver.destructor.Shake;
import solver.improve.Improver;

import java.util.ArrayList;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static util.ConcurrencyUtil.awaitAll;


@SuppressWarnings("DuplicatedCode")
public class ExchangerILS<I extends Instance, S extends Solution<I>> {

    private final int nRotateRounds;
    private final ILSConfig[] configs;

    /**
     * Create a new MultiStartAlgorithm, @see algorithm
     */
    public ExchangerILS(int nRotateRounds, ILSConfig... configs) {
        this.nRotateRounds = nRotateRounds;
        this.configs = configs;
    }

    public Result execute(Instance ins, int repetitions) {
        Result result = new Result(repetitions, this.toString(), ins.getName());
        for (int i = 0; i < repetitions; i++) {
            long startTime = System.nanoTime();
            Solution<I> s = algorithm(ins);
            long ellapsedTime = System.nanoTime() - startTime;
            result.addSolution(s, ellapsedTime);
        }

        return result;
    }

    /**
     * Executes the algorythm for the given instance
     * @param ins Instance the algorithm will process
     * @return Best solution found
     */
    public Solution<I> algorithm(Instance ins) {

        int nThreads = Runtime.getRuntime().availableProcessors() / 2;

        var nWorkers = this.configs.length;
        if(nThreads < nWorkers){
            System.out.format("[Warning] Available nThreads (%s) is less than the number of configured workers (%s), performance may be reduced\n", nThreads, nWorkers);
        }

        // Create threads and workers
        var executor = Executors.newFixedThreadPool(nThreads);
        var first = new LinkedBlockingQueue<Solution<I>>();
        var activeWorkers = new AtomicInteger(nWorkers);
        var barrier = new CyclicBarrier(nWorkers, () -> activeWorkers.set(nWorkers));
        var prev = first;
        var workers = new ArrayList<Worker>();
        for (int i = 0; i < nWorkers; i++) {
            // Next is a new queue if not the last element, and the first element if we are the last
            var next = i == nWorkers - 1? first: new LinkedBlockingQueue<Solution<I>>();
            workers.add(new Worker(executor, configs[i], prev, next, barrier, activeWorkers, nWorkers, nRotateRounds));
            prev = next;
        }

        // Generate initial solutions IN PARALLEL
        var futures = new ArrayList<Future<Solution<I>>>();
        for (var worker : workers) {
            futures.add(worker.buildInitialSolution(ins));
        }

        // Wait until all solutions are build before starting the next phase
        var solutions = awaitAll(futures);

        // Shuffle list and start working on them
        //Collections.shuffle(solutions, RandomManager.getRandom());

        // Improvement rounds
        futures = new ArrayList<>();
        for (int j = 0; j < workers.size(); j++) {
            futures.add(workers.get(j).startWorker(solutions.get(j)));
        }

        Solution<I> best = Solution.getBest(awaitAll(futures));
        executor.shutdown();

        return best;
    }

    public class ILSConfig {
        final Constructor<S> constructor;
        final ConstructiveNeighborhood constructiveNeighborhood;
        final SolutionBuilder solutionBuilder;
        final Shake shake;
        final Improver improver;
        final int shakeStrength;
        final int nShakes;

        public ILSConfig(int shakeStrength, int nShakes, Supplier<Constructor<S>> constructorSupplier, ConstructiveNeighborhood constructiveNeighborhood, SolutionBuilder solutionBuilder, Supplier<Shake> destructorSupplier, Supplier<Improver> improver) {
            this.shakeStrength = shakeStrength;
            this.nShakes = nShakes;
            this.constructiveNeighborhood = constructiveNeighborhood;
            this.solutionBuilder = solutionBuilder;
            assert constructorSupplier != null && destructorSupplier != null && improver != null;
            this.constructor = constructorSupplier.get();
            this.shake = destructorSupplier.get();
            this.improver = improver.get();
        }

        public ILSConfig(int shakeStrength, Supplier<Constructor<S>> constructorSupplier, ConstructiveNeighborhood constructiveNeighborhood, SolutionBuilder solutionBuilder, Supplier<Shake> destructorSupplier, Supplier<Improver> improver) {
            this(shakeStrength, -1, constructorSupplier, constructiveNeighborhood, solutionBuilder, destructorSupplier, improver);
        }

        @Override
        public String toString() {
            return "ILSConfig{" +
                    "shakeStrength=" + shakeStrength +
                    ", nShakes=" + nShakes +
                    ", constructor=" + constructor +
                    ", destructor=" + shake +
                    ", improver=" + improver +
                    '}';
        }
    }

    public class Worker {
        private final ExecutorService executor;
        private final ILSConfig config;
        private final BlockingQueue<Solution<I>> prev;
        private final BlockingQueue<Solution<I>> next;
        private final CyclicBarrier barrier;
        private final AtomicInteger activeWorkers; // Used to sync the workers
        private final int nWorkers;
        private final int nRotaterounds;

        public Worker(ExecutorService executor, ILSConfig config, BlockingQueue<Solution<I>> prev, BlockingQueue<Solution<I>> next, CyclicBarrier barrier, AtomicInteger activeWorkers, int nWorkers, int nRotaterounds) {
            this.executor = executor;
            this.config = config;
            this.prev = prev;
            this.next = next;
            this.activeWorkers = activeWorkers;
            this.barrier = barrier;
            this.nWorkers = nWorkers;
            this.nRotaterounds = nRotaterounds;
        }

        public Future<Solution<I>> buildInitialSolution(Instance instance){
            return executor.submit(() -> config.constructor.construct(instance, config.solutionBuilder, config.constructiveNeighborhood));
        }

        public Future<Solution<I>> startWorker(Solution<I> initialSolution){
            return executor.submit(() -> work(initialSolution));
        }

        private Solution<I> work(Solution<I> initialSolution) throws InterruptedException, BrokenBarrierException {
            Solution<I> best = initialSolution;
            var nShakes = this.config.nShakes;
            var iterations = nShakes / (this.nWorkers * nRotaterounds);

            for (int round = 0; round < nRotaterounds; round++) {
                // Do assigned work
                for (int i = 1; i <= iterations; i++) {
                    best = iteration(best);
                }

                // Mark current worker as finished
                activeWorkers.decrementAndGet();

                // Keep working until everyone finishes
                while(activeWorkers.get() != 0){
                    best = iteration(best);
                }

                next.add(best);                       // Push para el siguiente thread
                this.barrier.await();                 // Esperamos a que todos  hayan hecho push

                best = prev.take();                   // Pull del thread previo
            }

            return best;
        }

        private Solution<I> iteration(Solution<I> best) {
            var current = best.clone();
            this.config.shake.iteration(current, this.config.shakeStrength);
            this.config.improver.improve(current);
            if (current.getOptimalValue() < best.getOptimalValue()) {
                best = current;
            }
            return best;
        }
    }
}
