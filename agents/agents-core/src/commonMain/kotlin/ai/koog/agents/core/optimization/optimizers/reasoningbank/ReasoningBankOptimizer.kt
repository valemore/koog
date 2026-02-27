package ai.koog.agents.core.optimization.optimizers.reasoningbank

import ai.koog.agents.core.agent.GraphAIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.optimization.core.OptimizableNode
import ai.koog.agents.core.optimization.core.OptimizationConfig
import ai.koog.agents.core.optimization.features.CollectedTrajectory
import ai.koog.agents.core.optimization.features.TrajectoryCollectionFeature
import ai.koog.agents.core.optimization.features.collectTrajectory
import ai.koog.agents.core.optimization.optimizers.utils.findOptimizableNodes
import ai.koog.agents.core.optimization.optimizers.utils.prettyPrint
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.withContext

private val logger = KotlinLogging.logger {}

/**
 * Training example for ReasoningBank optimization.
 *
 * @param TInput The agent's input type.
 * @property input The input to pass to the agent.
 * @property query The query string used for memory retrieval and embedding (extracted from input).
 */
public data class ReasoningBankExample<TInput>(
    val input: TInput,
    val query: String,
)

/**
 * Result of ReasoningBank optimization.
 *
 * @property bank The trained reasoning bank containing memories.
 * @property targetNodeName The name of the optimizable node that receives memory context.
 * @property iterations Number of training examples processed.
 */
public data class ReasoningBankOptimizationResult(
    val bank: ReasoningBank,
    val targetNodeName: String,
    val iterations: Int,
)

/**
 * ReasoningBank optimizer.
 *
 * Extracts memories (lessons learned) from agent trajectories and stores them
 * in a memory bank with embeddings. At inference time, relevant memories are
 * retrieved via cosine similarity and injected into the agent's prompt.
 *
 * Unlike ACE's playbook (which is a static artifact), ReasoningBank performs
 * dynamic per-query retrieval. The optimized agent must do memory retrieval
 * at runtime before each execution.
 *
 * @property metricThreshold Minimum metric score to consider a run successful.
 * @property memoriesToRetrieve Number of memories to retrieve per query at inference time.
 * @property retrieveOnlySuccesses Whether to retrieve only memories from successful trajectories.
 */
public class ReasoningBankTrainer(
    public val metricThreshold: Double = 1.0,
    public val memoriesToRetrieve: Int = 5,
    public val retrieveOnlySuccesses: Boolean = false,
) {
    /**
     * Trains the reasoning bank by running the agent on training examples and
     * extracting memories from each trajectory.
     *
     * @param TInput The agent's input type.
     * @param TOutput The agent's output type.
     * @param promptExecutor The executor for LLM calls.
     * @param agentConfig The agent configuration.
     * @param strategy The agent strategy. Must contain at least one [OptimizableNode].
     * @param toolRegistry Tools available to the agent.
     * @param trainset Training examples.
     * @param metric Evaluation function: (query, agentOutput) -> score.
     * @param memoriesExtractorModel LLM model for extracting memories from trajectories.
     * @param bank The reasoning bank to populate. Defaults to empty.
     * @param targetNodeName Name of the optimizable node for memory injection. If null, uses first found.
     * @return [ReasoningBankOptimizationResult] containing the trained bank.
     */
    @OptIn(InternalAgentsApi::class)
    public suspend fun <TInput, TOutput> optimize(
        promptExecutor: PromptExecutor,
        agentConfig: AIAgentConfig,
        strategy: AIAgentGraphStrategy<TInput, TOutput>,
        toolRegistry: ToolRegistry = ToolRegistry.EMPTY,
        trainset: List<ReasoningBankExample<TInput>>,
        metric: (query: String, output: TOutput) -> Double,
        memoriesExtractorModel: LLModel,
        bank: ReasoningBank = ReasoningBank(TextEmbedder { error("TextEmbedder not provided") }),
        targetNodeName: String? = null,
    ): ReasoningBankOptimizationResult {
        require(trainset.isNotEmpty()) { "trainset is required for ReasoningBank" }

        val optimizableNodes = strategy.findOptimizableNodes()
        require(optimizableNodes.isNotEmpty()) { "Strategy must contain at least one OptimizableNode" }

        val targetNode = if (targetNodeName != null) {
            optimizableNodes.find { it.name == targetNodeName }
                ?: error("OptimizableNode '$targetNodeName' not found in strategy")
        } else {
            optimizableNodes.first()
        }

        for ((index, example) in trainset.withIndex()) {
            try {
                logger.info { "Processing item #${index + 1} / ${trainset.size}..." }

                // Run agent with trajectory collection and memory injection
                val memoryConfig = buildMemoryConfig(
                    targetNode = targetNode,
                    bank = bank,
                    query = example.query,
                )

                val tracingAgent = GraphAIAgent(
                    inputType = strategy.inputType,
                    outputType = strategy.outputType,
                    promptExecutor = promptExecutor,
                    agentConfig = agentConfig,
                    strategy = strategy,
                    toolRegistry = toolRegistry,
                    installFeatures = { collectTrajectory() },
                )

                val pipeline = tracingAgent.createSession().pipeline()
                    ?: error("Pipeline should be available after createSession()")
                val trajectory = pipeline.feature(CollectedTrajectory::class, TrajectoryCollectionFeature)
                    ?: error("TrajectoryCollectionFeature should have been installed")

                val output = withContext(memoryConfig) { tracingAgent.run(example.input) }

                val score = metric(example.query, output)
                val isSuccess = score >= metricThreshold

                // Extract memories from trajectory
                val lessons = extractMemoryItems(
                    query = example.query,
                    isSuccess = isSuccess,
                    trajectory = trajectory,
                    promptExecutor = promptExecutor,
                    model = memoriesExtractorModel,
                )
                lessons.forEach { lesson -> bank.addMemory(lesson) }
            } catch (e: Exception) {
                logger.error(e) { "Failed to process item; skipping it. Reason:\n${e.stackTraceToString()}" }
            }
        }

        logger.info { "ReasoningBank optimization completed." }
        return ReasoningBankOptimizationResult(
            bank = bank,
            targetNodeName = targetNode.name,
            iterations = trainset.size,
        )
    }

    /**
     * Builds an [OptimizationConfig] that injects retrieved memory context into the target node.
     */
    public suspend fun buildMemoryConfig(
        targetNode: OptimizableNode<*, *>,
        bank: ReasoningBank,
        query: String,
    ): OptimizationConfig {
        val memories = bank.retrieve(query, memoriesToRetrieve, retrieveOnlySuccesses)
        val memoryContext = ReasoningBankPrompts.constructMemoryContext(memories)
        val fullInstruction = if (memoryContext.isNotBlank()) {
            targetNode.instruction + "\n\n" + memoryContext
        } else {
            targetNode.instruction
        }
        return OptimizationConfig(
            instructions = mapOf(targetNode.name to fullInstruction),
        )
    }

    private suspend fun extractMemoryItems(
        query: String,
        isSuccess: Boolean,
        trajectory: CollectedTrajectory,
        promptExecutor: PromptExecutor,
        model: LLModel,
    ): List<MemoryItem> {
        val fullPrompt = trajectory.getLatestFullPrompt()
        if (fullPrompt == null) {
            logger.warn { "No trajectory captured (no LLM calls made). Skipping memory extraction." }
            return emptyList()
        }

        val systemPrompt = if (isSuccess) {
            ReasoningBankPrompts.EXTRACT_MEMORY_ITEMS_SUCCESS_SYSTEM
        } else {
            ReasoningBankPrompts.EXTRACT_MEMORY_ITEMS_FAILURE_SYSTEM
        }
        val memoryType = if (isSuccess) MemoryType.FROM_SUCCESS else MemoryType.FROM_FAILURE

        val extractionPrompt = prompt("extract-memory-items") {
            system(systemPrompt)
            user("Query: $query\n\nTrajectory:\n${fullPrompt.prettyPrint()}")
        }

        val result = try {
            promptExecutor.execute(extractionPrompt, model).first().content
        } catch (e: Exception) {
            logger.warn { "Failed to extract memory items: ${e.message}" }
            return emptyList()
        }

        return parseRawMemoryItems(result).map { raw ->
            MemoryItem(
                query = query,
                title = raw.title,
                description = raw.description,
                content = raw.content,
                type = memoryType,
                id = generateMemoryId(),
            )
        }
    }

    private var memoryIdCounter = 0

    private fun generateMemoryId(): String = "mem-${memoryIdCounter++}"
}

private data class RawMemoryItem(
    val title: String,
    val description: String,
    val content: String,
)

private fun parseRawMemoryItems(input: String): List<RawMemoryItem> {
    // Use [\s\S] instead of . for cross-platform multiline matching (DOT_MATCHES_ALL is JVM-only)
    val regex = Regex(
        pattern = """# Memory Item[\s\S]*?\s+## Title\s+([\s\S]*?)\s+## Description\s+([\s\S]*?)\s+## Content\s+([\s\S]*?)(?=\s*# Memory Item|$)""",
    )

    fun String.stripBackticks(): String =
        this.trim().removePrefix("```").removeSuffix("```").trim()

    return regex.findAll(input).map { matchResult ->
        val (title, description, content) = matchResult.destructured
        RawMemoryItem(
            title = title.trim().stripBackticks(),
            description = description.trim().stripBackticks(),
            content = content.trim().stripBackticks()
        )
    }.toList()
}
