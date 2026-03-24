package ai.koog.agents.core.optimization.core

import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.agent.entity.createStorageKey
import kotlinx.serialization.Serializable

/**
 * Immutable configuration holding optimized instructions and demonstrations for a strategy.
 *
 * Produced by optimizers and consumed by optimizable subgraphs at runtime.
 * Can be saved to and loaded from the filesystem for persistent optimization results.
 *
 * Immutability simplifies parallel evaluation: each optimization trial can safely hold
 * its own config instance without synchronization.
 *
 * Subgraph entries use path-based keys for nested subgraph support (e.g. "outer/inner").
 * The strategy-level instruction and demonstrations are stored separately from subgraph entries.
 *
 * @property strategyInstruction Optimized instruction for the strategy's system prompt, or null to keep the original.
 * @property strategyDemonstrations Demonstrations for the strategy level.
 * @property subgraphInstructions Map from subgraph path to optimized instruction text.
 * @property subgraphDemonstrations Map from subgraph path to demonstration list.
 */
@Serializable
public data class OptimizationConfig(
    val strategyInstruction: String? = null,
    val strategyDemonstrations: List<Demonstration> = emptyList(),
    val subgraphInstructions: Map<String, String> = emptyMap(),
    val subgraphDemonstrations: Map<String, List<Demonstration>> = emptyMap(),
) {
    /**
     * Returns the optimized instruction for the given subgraph path, or null if not configured.
     */
    public fun getInstruction(subgraphPath: String): String? = subgraphInstructions[subgraphPath]

    /**
     * Returns the demonstrations for the given subgraph path, or an empty list if not configured.
     */
    public fun getDemonstrations(subgraphPath: String): List<Demonstration> =
        subgraphDemonstrations[subgraphPath].orEmpty()

    /**
     * Creates a new config with an additional or updated subgraph instruction.
     */
    public fun withSubgraphInstruction(subgraphPath: String, instruction: String): OptimizationConfig =
        copy(subgraphInstructions = subgraphInstructions + (subgraphPath to instruction))

    /**
     * Creates a new config with additional or updated subgraph demonstrations.
     */
    public fun withSubgraphDemonstrations(
        subgraphPath: String,
        demonstrations: List<Demonstration>,
    ): OptimizationConfig =
        copy(subgraphDemonstrations = subgraphDemonstrations + (subgraphPath to demonstrations))

    /**
     * Creates a new config with an updated strategy-level instruction.
     */
    public fun withStrategyInstruction(instruction: String): OptimizationConfig =
        copy(strategyInstruction = instruction)

    /**
     * Creates a new config with updated strategy-level demonstrations.
     */
    public fun withStrategyDemonstrations(demonstrations: List<Demonstration>): OptimizationConfig =
        copy(strategyDemonstrations = demonstrations)

    /**
     * Merges this config with another, with the [other] config's entries taking precedence.
     */
    public fun mergeWith(other: OptimizationConfig): OptimizationConfig = OptimizationConfig(
        strategyInstruction = other.strategyInstruction ?: strategyInstruction,
        strategyDemonstrations = other.strategyDemonstrations.ifEmpty { strategyDemonstrations },
        subgraphInstructions = subgraphInstructions + other.subgraphInstructions,
        subgraphDemonstrations = subgraphDemonstrations + other.subgraphDemonstrations,
    )

    public companion object {
        /** Storage key for injecting the optimization config into agent storage at runtime. */
        public val STORAGE_KEY: AIAgentStorageKey<OptimizationConfig> =
            createStorageKey("optimization-config")
    }
}
