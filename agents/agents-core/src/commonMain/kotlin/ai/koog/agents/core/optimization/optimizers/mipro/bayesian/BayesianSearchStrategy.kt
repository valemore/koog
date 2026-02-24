package ai.koog.agents.core.optimization.optimizers.mipro.bayesian

import ai.koog.agents.core.optimization.core.Demonstration
import ai.koog.agents.core.optimization.core.OptimizationConfig
import ai.koog.agents.core.optimization.core.OptimizationResult
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlin.random.Random

private val logger = KotlinLogging.logger {}

/**
 * Bayesian search over instruction/demo combinations using TPE.
 *
 * Mirrors the behavior of `randomGridSearch` in MIPROv2:
 * baseline evaluation -> trials -> periodic full evaluation -> final full evaluation.
 *
 * The difference is that instead of sampling configurations uniformly at random,
 * the TPE sampler learns from past evaluations to focus on promising configurations.
 *
 * @param moduleNames Names of the optimizable nodes.
 * @param instructionCandidates Map from the module name to a list of candidate instructions.
 * @param demoCandidates Map from the module name to a list of candidate demo sets, or null for zero-shot.
 * @param defaultInstructions Map from module name to default instruction (used when no candidates exist).
 * @param numTrials Number of search trials to run.
 * @param tpeConfig TPE algorithm configuration.
 * @param minibatch Whether to use minibatch evaluation.
 * @param minibatchFullEvalSteps How often to run a full evaluation during a minibatch search.
 * @param random Random source for reproducibility.
 * @param evaluate Evaluates a configuration and returns a score. The `fullEval` parameter indicates
 *  whether to evaluate on the full validation set (true) or a minibatch (false).
 * @return The best [OptimizationResult] found.
 */
internal suspend fun bayesianSearch(
    moduleNames: List<String>,
    instructionCandidates: Map<String, List<String>>,
    demoCandidates: Map<String, List<List<Demonstration<*, *>>>>?,
    defaultInstructions: Map<String, String>,
    numTrials: Int,
    tpeConfig: TPEConfig,
    minibatch: Boolean,
    minibatchFullEvalSteps: Int,
    random: Random,
    evaluate: suspend (config: OptimizationConfig, fullEval: Boolean) -> Double,
): OptimizationResult {
    // Build dimensions from candidate counts
    val dimensions = buildDimensions(moduleNames, instructionCandidates, demoCandidates)
    val sampler = CategoricalTPESampler(dimensions, tpeConfig)

    // Evaluate baseline (empty config)
    logger.info { "Evaluating baseline..." }
    val baselineConfig = OptimizationConfig()
    val baselineScore = evaluate(baselineConfig, true)

    var bestScore = baselineScore
    var bestConfig = baselineConfig
    logger.info { "Baseline score: ${fmt(baselineScore)}" }

    val observations = mutableListOf<Observation>()

    for (trial in 1..numTrials) {
        // Suggest a configuration
        val indices = sampler.suggest(observations, random)

        // Map indices back to actual instructions and demos
        val trialConfig = indicesToConfig(
            indices = indices,
            moduleNames = moduleNames,
            instructionCandidates = instructionCandidates,
            demoCandidates = demoCandidates,
            defaultInstructions = defaultInstructions,
        )

        // Evaluate on a validation set or minibatch
        val score = evaluate(trialConfig, !minibatch)

        // Record observation for TPE
        observations.add(Observation(config = indices, score = score))

        if (score > bestScore) {
            logger.info { "Trial $trial/$numTrials: new best ${fmt(score)} (was ${fmt(bestScore)})" }
            bestScore = score
            bestConfig = trialConfig
        } else {
            logger.info { "Trial $trial/$numTrials: ${fmt(score)} (best=${fmt(bestScore)})" }
        }

        // Periodic full evaluation when using minibatch
        if (minibatch && trial % minibatchFullEvalSteps == 0) {
            val fullScore = evaluate(bestConfig, true)
            logger.info { "  Full eval at trial $trial: ${fmt(fullScore)}" }
        }
    }

    // Final full evaluation of the best config
    val finalScore = evaluate(bestConfig, true)

    return OptimizationResult(
        config = bestConfig,
        score = finalScore,
        iterations = numTrials,
        metadata = mapOf(
            "optimizer" to "MIPROv2-TPE",
            "baselineScore" to baselineScore,
            "numTrials" to numTrials,
            "numModules" to moduleNames.size,
        ),
    )
}

/**
 * Builds the TPE dimensions from the candidate counts.
 *
 * For each module, creates an instruction dimension (if there are candidates)
 * and a demo dimension (if demo candidates exist).
 */
private fun buildDimensions(
    moduleNames: List<String>,
    instructionCandidates: Map<String, List<String>>,
    demoCandidates: Map<String, List<List<Demonstration<*, *>>>>?,
): List<CategoricalDimension> {
    val dims = mutableListOf<CategoricalDimension>()

    for (name in moduleNames) {
        val numInstructions = instructionCandidates[name]?.size ?: 0
        if (numInstructions > 0) {
            dims.add(CategoricalDimension("${name}_instruction", numInstructions))
        }

        if (demoCandidates != null) {
            val numDemos = demoCandidates[name]?.size ?: 0
            if (numDemos > 0) {
                dims.add(CategoricalDimension("${name}_demos", numDemos))
            }
        }
    }

    // If no dimensions were created (no candidates for anything), add a dummy dimension
    // so the sampler doesn't fail on empty dimensions.
    if (dims.isEmpty()) {
        dims.add(CategoricalDimension("_dummy", 1))
    }

    return dims
}

/**
 * Converts the TPE sampler index output back to an [OptimizationConfig] with actual
 * instruction strings and demo lists.
 */
private fun indicesToConfig(
    indices: Map<String, Int>,
    moduleNames: List<String>,
    instructionCandidates: Map<String, List<String>>,
    demoCandidates: Map<String, List<List<Demonstration<*, *>>>>?,
    defaultInstructions: Map<String, String>,
): OptimizationConfig {
    val instructions = moduleNames.associateWith { name ->
        val candidates = instructionCandidates[name]
        val idx = indices["${name}_instruction"]
        val instruction = if (candidates != null && idx != null && idx in candidates.indices) {
            candidates[idx]
        } else {
            defaultInstructions[name] ?: ""
        }
        instruction
    }

    val demonstrations: Map<String, List<Demonstration<*, *>>> = if (demoCandidates != null) {
        moduleNames.associateWith { name ->
            val candidates = demoCandidates[name]
            val idx = indices["${name}_demos"]
            val demos = if (candidates != null && idx != null && idx in candidates.indices) {
                candidates[idx]
            } else {
                emptyList()
            }
            demos
        }
    } else {
        emptyMap()
    }

    return OptimizationConfig(
        instructions = instructions,
        demonstrations = demonstrations,
    )
}

/** Format a Double to 3 decimal places (commonMain-compatible). */
private fun fmt(d: Double): String {
    val whole = d.toLong()
    val frac = ((d - whole) * 1000 + 0.5).toLong()
    return "$whole.${frac.toString().padStart(3, '0')}"
}
