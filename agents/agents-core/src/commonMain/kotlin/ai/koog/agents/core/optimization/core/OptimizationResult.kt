package ai.koog.agents.core.optimization.core

import kotlinx.serialization.Serializable

/**
 * The result of an optimization run.
 *
 * Contains the optimized [config] that can be applied to an agent, along with
 * metadata about the optimization process.
 *
 * @property config The optimized configuration with tuned instructions and bootstrapped demonstrations.
 * @property score The best evaluation score achieved during optimization.
 * @property iterations The number of training examples processed.
 * @property metadata Additional optimizer-specific information (e.g. optimizer name, parameters).
 */
@Serializable
public data class OptimizationResult(
    val config: OptimizationConfig,
    val score: Double,
    val iterations: Int,
    val metadata: Map<String, String> = emptyMap(),
)
