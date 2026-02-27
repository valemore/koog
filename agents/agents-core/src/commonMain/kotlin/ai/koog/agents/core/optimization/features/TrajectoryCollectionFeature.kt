package ai.koog.agents.core.optimization.features

import ai.koog.agents.core.agent.GraphAIAgent.FeatureContext
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.feature.AIAgentGraphFeature
import ai.koog.agents.core.feature.config.FeatureConfig
import ai.koog.agents.core.feature.pipeline.AIAgentGraphPipeline
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.message.Message
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Configuration for the [TrajectoryCollectionFeature].
 */
public class TrajectoryCollectionConfig : FeatureConfig()

/**
 * Collected full conversation trajectory from an agent run.
 *
 * After each LLM call, the prompt (containing conversation history) and the LLM responses
 * are captured. The [getLatestFullPrompt] reflects the last captured state — typically the
 * complete conversation history at the end of the agent run.
 */
public class CollectedTrajectory {
    private val mutex = Mutex()

    private var _latestFullPrompt: Prompt? = null

    /** Returns the most recent full prompt including all LLM responses, or null if no LLM call was made. */
    public suspend fun getLatestFullPrompt(): Prompt? = mutex.withLock { _latestFullPrompt }

    internal suspend fun recordLLMCall(prompt: Prompt, responses: List<Message.Response>) {
        mutex.withLock {
            _latestFullPrompt = prompt(prompt) { messages(responses) }
        }
    }
}

/**
 * Feature that captures the full conversation trajectory during agent execution.
 *
 * Intercepts `LLMCallCompleted` pipeline events to record the prompt and responses after
 * each LLM call. Used by trajectory-based optimizers (ACE, ReasoningBank) that need the
 * complete conversation history for analysis.
 *
 * Example usage:
 * ```kotlin
 * val agent = GraphAIAgent(strategy = myStrategy, installFeatures = {
 *     collectTrajectory()
 * })
 * val pipeline = agent.createSession().pipeline()
 * val trajectory = pipeline?.feature(CollectedTrajectory::class, TrajectoryCollectionFeature)
 * val fullPrompt = trajectory?.getLatestFullPrompt()
 * ```
 */
public object TrajectoryCollectionFeature : AIAgentGraphFeature<TrajectoryCollectionConfig, CollectedTrajectory> {

    override val key: AIAgentStorageKey<CollectedTrajectory> =
        AIAgentStorageKey("optimization-trajectory-collection")

    override fun createInitialConfig(): TrajectoryCollectionConfig = TrajectoryCollectionConfig()

    override fun install(config: TrajectoryCollectionConfig, pipeline: AIAgentGraphPipeline): CollectedTrajectory {
        val trajectory = CollectedTrajectory()

        pipeline.interceptLLMCallCompleted(this) { eventContext ->
            trajectory.recordLLMCall(eventContext.prompt, eventContext.responses)
        }

        return trajectory
    }
}

/**
 * Installs the [TrajectoryCollectionFeature] to capture the full conversation trajectory.
 */
public fun FeatureContext.collectTrajectory(configure: TrajectoryCollectionConfig.() -> Unit = {}) {
    install(TrajectoryCollectionFeature) {
        configure()
    }
}
