package ai.koog.agents.core.optimization.features

import ai.koog.agents.core.agent.GraphAIAgent.FeatureContext
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.feature.AIAgentGraphFeature
import ai.koog.agents.core.feature.config.FeatureConfig
import ai.koog.agents.core.feature.pipeline.AIAgentGraphPipeline
import ai.koog.agents.core.optimization.core.Demonstration
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.message.Message
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Returns a storage key for intermediate messages captured by an optimizable subgraph.
 *
 * The `afterFinishToolCall` hook in each optimizable subgraph writes its prompt messages
 * to this key before the prompt is discarded on freshHistory restore. The
 * [SubgraphTraceCollectionFeature] reads this key in its subgraph completion handler
 * to populate [Demonstration.intermediateMessages].
 *
 * Safe for concurrent use: [AIAgentStorageKey] implements value-based equality on [name],
 * so independently created keys with the same subgraph name resolve to the same storage entry.
 *
 * @param subgraphName The name of the subgraph whose intermediate messages are stored.
 * @return A storage key for the intermediate messages list.
 */
public fun intermediateMessagesKey(subgraphName: String): AIAgentStorageKey<List<Message>> =
    createStorageKey("optimization-intermediate-messages-$subgraphName")

/**
 * Returns a storage key for the inherited prompt messages captured at the start of an
 * optimizable subgraph, before any subgraph-specific content is added.
 *
 * Used to compute the diff between inherited and final messages, so that only the
 * subgraph's own conversation is stored as intermediate messages (excluding parent history
 * inherited via `freshHistory = false`).
 *
 * @param subgraphName The name of the subgraph.
 * @return A storage key for the inherited messages list.
 */
public fun inheritedMessagesKey(subgraphName: String): AIAgentStorageKey<List<Message>> =
    createStorageKey("optimization-inherited-messages-$subgraphName")

/**
 * Configuration for [SubgraphTraceCollectionFeature].
 *
 * @property maxTracesPerSubgraph Maximum traces to store per subgraph. Zero or negative means unlimited.
 * @property stringifyInput Converts subgraph input to string for demonstration storage.
 * @property stringifyOutput Converts subgraph output to string for demonstration storage.
 */
public class SubgraphTraceCollectionConfig : FeatureConfig() {
    /** Maximum traces per subgraph. Zero or negative means unlimited. */
    public var maxTracesPerSubgraph: Int = 100

    /** Converts subgraph input to string for demonstration storage. */
    public var stringifyInput: (Any?) -> String = { it.toString() }

    /** Converts subgraph output to string for demonstration storage. */
    public var stringifyOutput: (Any?) -> String = { it.toString() }
}

/**
 * Thread-safe container for collected subgraph execution traces and whole-agent trajectory.
 *
 * Stores per-subgraph [Demonstration] objects keyed by subgraph name, and the latest
 * whole-agent conversation trajectory. Safe for concurrent access via internal mutex.
 */
public class CollectedSubgraphTraces(
    private val maxPerSubgraph: Int,
) {
    private val traces = mutableMapOf<String, MutableList<Demonstration>>()
    private var latestFullPrompt: Prompt? = null
    private val mutex = Mutex()

    /**
     * Adds a demonstration for a subgraph.
     */
    internal suspend fun addTrace(subgraphName: String, demonstration: Demonstration) {
        mutex.withLock {
            val subgraphTraces = traces.getOrPut(subgraphName) { mutableListOf() }
            if (maxPerSubgraph > 0 && subgraphTraces.size >= maxPerSubgraph) {
                subgraphTraces.removeAt(0)
            }
            subgraphTraces.add(demonstration)
        }
    }

    /**
     * Records an LLM call to build the whole-agent trajectory.
     * Called after every LLM call; the latest state reflects the full conversation.
     */
    internal suspend fun recordLLMCall(currentPrompt: Prompt, responses: List<Message.Response>) {
        mutex.withLock {
            latestFullPrompt = prompt(currentPrompt) { messages(responses) }
        }
    }

    /**
     * Returns all collected traces for a specific subgraph.
     */
    public suspend fun getTraces(subgraphName: String): List<Demonstration> =
        mutex.withLock {
            traces[subgraphName]?.toList().orEmpty()
        }

    /**
     * Returns all collected traces grouped by subgraph name.
     */
    public suspend fun getAllTraces(): Map<String, List<Demonstration>> =
        mutex.withLock {
            traces.mapValues { (_, v) -> v.toList() }
        }

    /**
     * Returns the latest whole-agent conversation trajectory, or null if no LLM call was made.
     *
     * This is the full prompt (including all LLM responses) captured after the last LLM call
     * in the agent run. Useful for trajectory-based optimizers (ACE, ReasoningBank).
     */
    public suspend fun getLatestFullPrompt(): Prompt? =
        mutex.withLock { latestFullPrompt }

    /**
     * Clears all collected traces and the trajectory.
     */
    public suspend fun clear() {
        mutex.withLock {
            traces.clear()
            latestFullPrompt = null
        }
    }
}

/**
 * `Optimizer-internal` feature that captures both per-subgraph traces and whole-agent trajectory.
 *
 * Installed by optimizers on temporary agents during training to collect demonstrations
 * from successful runs. Not intended for direct end-user use.
 *
 * Collects two kinds of data:
 * - **Per-subgraph traces**: [Demonstration] objects with input, output, and optional
 *   intermediate messages. Intermediate messages are exported by optimizable subgraphs
 *   to storage via [intermediateMessagesKey] before freshHistory discards the prompt.
 * - **Whole-agent trajectory**: the full conversation prompt captured after each LLM call
 *   via [interceptLLMCallCompleted]. The latest state is the complete conversation at
 *   the end of the run. Useful for ACE/ReasoningBank-style optimizers.
 */
public object SubgraphTraceCollectionFeature :
    AIAgentGraphFeature<SubgraphTraceCollectionConfig, CollectedSubgraphTraces> {

    override val key: AIAgentStorageKey<CollectedSubgraphTraces> =
        AIAgentStorageKey("optimization-subgraph-trace-collection")

    override fun createInitialConfig(): SubgraphTraceCollectionConfig =
        SubgraphTraceCollectionConfig()

    override fun install(
        config: SubgraphTraceCollectionConfig,
        pipeline: AIAgentGraphPipeline,
    ): CollectedSubgraphTraces {
        val collected = CollectedSubgraphTraces(config.maxTracesPerSubgraph)

        // Per-subgraph trace collection
        pipeline.interceptSubgraphExecutionCompleted(this) { eventContext ->
            val name = eventContext.subgraph.name
            val intermediate = eventContext.context.storage.get(intermediateMessagesKey(name))

            val demo = Demonstration(
                input = config.stringifyInput(eventContext.input),
                output = config.stringifyOutput(eventContext.output),
                intermediateMessages = intermediate,
            )
            collected.addTrace(name, demo)
        }

        // Whole-agent trajectory: record prompt + responses after every LLM call
        pipeline.interceptLLMCallCompleted(this) { eventContext ->
            collected.recordLLMCall(eventContext.prompt, eventContext.responses)
        }

        return collected
    }
}

/**
 * Installs the [SubgraphTraceCollectionFeature] to collect subgraph execution traces
 * and whole-agent trajectory.
 *
 * @param configure Lambda to customize trace collection behavior.
 */
public fun FeatureContext.collectSubgraphTraces(
    configure: SubgraphTraceCollectionConfig.() -> Unit = {},
) {
    install(SubgraphTraceCollectionFeature) {
        configure()
    }
}
