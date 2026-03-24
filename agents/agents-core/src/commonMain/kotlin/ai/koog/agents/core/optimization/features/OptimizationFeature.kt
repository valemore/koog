package ai.koog.agents.core.optimization.features

import ai.koog.agents.core.agent.GraphAIAgent.FeatureContext
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.feature.AIAgentGraphFeature
import ai.koog.agents.core.feature.config.FeatureConfig
import ai.koog.agents.core.feature.pipeline.AIAgentGraphPipeline
import ai.koog.agents.core.optimization.core.OptimizationConfig

/**
 * Configuration for the [OptimizationFeature].
 *
 * @property config The optimization config to inject into agent storage at runtime.
 */
public class OptimizationFeatureConfig : FeatureConfig() {
    /** The optimization config to make available to optimizable subgraphs. */
    public var config: OptimizationConfig = OptimizationConfig()
}

/**
 * Feature that injects an [OptimizationConfig] into agent storage at strategy start.
 *
 * Optimizable subgraphs read the config from storage to resolve their effective
 * instructions and demonstrations. Without this feature installed, optimizable subgraphs
 * fall back to their default instructions and empty demonstrations.
 *
 * Usage:
 * ```kotlin
 * val agent = AIAgent(...) {
 *     installOptimization {
 *         config = loadedConfig
 *     }
 * }
 * ```
 */
public object OptimizationFeature :
    AIAgentGraphFeature<OptimizationFeatureConfig, OptimizationConfig> {

    override val key: AIAgentStorageKey<OptimizationConfig> = OptimizationConfig.STORAGE_KEY

    override fun createInitialConfig(): OptimizationFeatureConfig = OptimizationFeatureConfig()

    override fun install(
        config: OptimizationFeatureConfig,
        pipeline: AIAgentGraphPipeline,
    ): OptimizationConfig {
        val optimizationConfig = config.config

        pipeline.interceptStrategyStarting(this) { eventContext ->
            eventContext.context.storage.set(OptimizationConfig.STORAGE_KEY, optimizationConfig)
        }

        return optimizationConfig
    }
}

/**
 * Installs the [OptimizationFeature] with the given configuration.
 *
 * This makes the [OptimizationConfig] available to all optimizable subgraphs in the strategy
 * via agent storage. Without this feature, optimizable subgraphs use their default instructions
 * and empty demonstrations.
 *
 * @param configure Lambda to set the optimization config.
 */
public fun FeatureContext.installOptimization(configure: OptimizationFeatureConfig.() -> Unit = {}) {
    install(OptimizationFeature) {
        configure()
    }
}
