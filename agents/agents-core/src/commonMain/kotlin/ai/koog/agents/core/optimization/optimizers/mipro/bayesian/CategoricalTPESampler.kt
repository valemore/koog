package ai.koog.agents.core.optimization.optimizers.mipro.bayesian

import kotlin.random.Random

/**
 * One axis of the categorical search space.
 *
 * @property name Unique identifier for this dimension (e.g., "thinking_instruction").
 * @property numCategories Number of distinct categories (indexed 0 until numCategories).
 */
public data class CategoricalDimension(
    val name: String,
    val numCategories: Int,
) {
    init {
        require(numCategories > 0) { "numCategories must be positive, got $numCategories for dimension '$name'" }
    }
}

/**
 * A completed trial: a full configuration with its observed score.
 *
 * @property config Map from dimension name to selected category index.
 * @property score The evaluation score for this configuration (higher is better).
 */
public data class Observation(
    val config: Map<String, Int>,
    val score: Double,
)

/**
 * Tuning knobs for the TPE algorithm.
 *
 * @property gamma Fraction of observations considered "good" (top quantile). Must be in (0, 1).
 * @property priorWeight Laplace smoothing weight added to each category count. Must be positive.
 * @property numCandidates Number of candidate configs sampled from l(x) per suggestion. Must be positive.
 * @property warmupTrials Number of observations before switching from uniform random to TPE.
 *  If null, defaults to `max(5, numDimensions + 1)`.
 */
public data class TPEConfig(
    val gamma: Double = 0.25,
    val priorWeight: Double = 1.0,
    val numCandidates: Int = 24,
    val warmupTrials: Int? = null,
) {
    init {
        require(gamma > 0.0 && gamma < 1.0) { "gamma must be in (0, 1), got $gamma" }
        require(priorWeight > 0.0) { "priorWeight must be positive, got $priorWeight" }
        require(numCandidates > 0) { "numCandidates must be positive, got $numCandidates" }
        if (warmupTrials != null) {
            require(warmupTrials >= 0) { "warmupTrials must be non-negative, got $warmupTrials" }
        }
    }
}

/**
 * Tree-structured Parzen Estimator (TPE) for purely categorical search spaces.
 *
 * For categorical variables, the TPE density estimation simplifies to counting
 * category frequencies with Laplace smoothing -- no Gaussian kernels or matrix
 * operations are needed.
 *
 * Algorithm:
 * 1. **Warmup**: While fewer than [warmupTrials] observations exist, sample uniformly at random.
 * 2. **Split**: Partition observations into "good" (top gamma quantile by score) and "bad" (rest).
 * 3. **Estimate**: Per dimension, count category frequencies in each group and add Laplace smoothing
 *    to produce probability distributions l(x) (good) and g(x) (bad).
 * 4. **Sample**: Draw [TPEConfig.numCandidates] configs from l(x), then pick the one maximizing
 *    the Expected Improvement proxy score: product of l(x_d) / g(x_d) across all dimensions d.
 *
 * @param dimensions The search space definition.
 * @param config TPE hyperparameters.
 */
public class CategoricalTPESampler(
    private val dimensions: List<CategoricalDimension>,
    private val config: TPEConfig = TPEConfig(),
) {
    init {
        require(dimensions.isNotEmpty()) { "dimensions must not be empty" }
        val names = dimensions.map { it.name }
        require(names.distinct().size == names.size) {
            "Duplicate dimension names: ${names.groupBy { it }.filter { it.value.size > 1 }.keys}"
        }
    }

    /** Effective warmup threshold, using default if not explicitly set. */
    private val warmupTrials: Int = config.warmupTrials ?: maxOf(5, dimensions.size + 1)

    /**
     * Whether the sampler is still in warmup phase (uniform random sampling).
     */
    public fun isWarmingUp(numObservations: Int): Boolean = numObservations < warmupTrials

    /**
     * Suggests a new configuration to evaluate.
     *
     * During warmup, returns a uniformly random configuration. After warmup,
     * uses the TPE algorithm to suggest a promising configuration.
     *
     * @param observations All completed trials so far.
     * @param random Random source for reproducibility.
     * @return Map from dimension name to category index.
     */
    public fun suggest(observations: List<Observation>, random: Random): Map<String, Int> {
        if (observations.isEmpty() || isWarmingUp(observations.size)) {
            return sampleUniform(random)
        }

        val (good, bad) = splitObservations(observations)

        // Build per-dimension distributions for good and bad groups
        val goodDists = dimensions.associate { dim ->
            dim.name to estimateCategoricalDistribution(dim, good)
        }
        val badDists = dimensions.associate { dim ->
            dim.name to estimateCategoricalDistribution(dim, bad)
        }

        // Sample numCandidates configs from l(x) and pick the best EI score
        var bestCandidate: Map<String, Int>? = null
        var bestEI = Double.NEGATIVE_INFINITY

        repeat(config.numCandidates) {
            val candidate = dimensions.associate { dim ->
                dim.name to sampleFromDistribution(goodDists[dim.name]!!, random)
            }
            val ei = computeEIScore(candidate, goodDists, badDists)
            if (ei > bestEI) {
                bestEI = ei
                bestCandidate = candidate
            }
        }

        return bestCandidate!!
    }

    /**
     * Samples a uniformly random configuration.
     */
    private fun sampleUniform(random: Random): Map<String, Int> {
        return dimensions.associate { dim ->
            dim.name to random.nextInt(dim.numCategories)
        }
    }

    /**
     * Splits observations into good (top gamma quantile) and bad (rest) by score.
     *
     * Guarantees at least 1 observation in each group: if the gamma split would leave
     * a group empty, adjusts the split point.
     */
    internal fun splitObservations(observations: List<Observation>): Pair<List<Observation>, List<Observation>> {
        require(observations.isNotEmpty()) { "Cannot split empty observations" }

        val sorted = observations.sortedByDescending { it.score }
        val numGood = maxOf(1, minOf(observations.size - 1, (observations.size * config.gamma).toInt()))

        val good = sorted.take(numGood)
        val bad = sorted.drop(numGood)

        return good to bad
    }

    /**
     * Estimates a categorical probability distribution from a group of observations
     * using frequency counting with Laplace smoothing.
     *
     * @param dim The dimension to estimate for.
     * @param group The subset of observations (good or bad).
     * @return Array of probabilities, one per category, summing to 1.0.
     */
    internal fun estimateCategoricalDistribution(
        dim: CategoricalDimension,
        group: List<Observation>,
    ): DoubleArray {
        val counts = DoubleArray(dim.numCategories) { config.priorWeight }

        for (obs in group) {
            val idx = obs.config[dim.name]
            if (idx != null && idx in 0 until dim.numCategories) {
                counts[idx] += 1.0
            }
        }

        val total = counts.sum()
        return DoubleArray(dim.numCategories) { counts[it] / total }
    }

    /**
     * Samples a category index from a probability distribution using inverse CDF.
     */
    private fun sampleFromDistribution(probs: DoubleArray, random: Random): Int {
        val u = random.nextDouble()
        var cumulative = 0.0
        for (i in probs.indices) {
            cumulative += probs[i]
            if (u < cumulative) return i
        }
        return probs.size - 1
    }

    /**
     * Computes the Expected Improvement proxy score for a candidate configuration.
     *
     * EI(x) ~ product over dimensions d of: l(x_d) / g(x_d)
     *
     * Uses log-space to avoid underflow with many dimensions.
     */
    internal fun computeEIScore(
        candidate: Map<String, Int>,
        goodDists: Map<String, DoubleArray>,
        badDists: Map<String, DoubleArray>,
    ): Double {
        var logEI = 0.0
        for (dim in dimensions) {
            val idx = candidate[dim.name] ?: continue
            val lx = goodDists[dim.name]!![idx]
            val gx = badDists[dim.name]!![idx]
            // Both lx and gx are > 0 due to Laplace smoothing
            logEI += kotlin.math.ln(lx) - kotlin.math.ln(gx)
        }
        return logEI
    }
}
