package ai.koog.agents.core.optimization.optimizers.ace

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
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.structure.executeStructured
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.withContext

private val logger = KotlinLogging.logger {}

/**
 * Training example for ACE optimization.
 *
 * @param TInput The agent's input type.
 * @property input The input to pass to the agent.
 * @property label The ground truth label, used by the reflector to analyze trajectory success/failure.
 */
public data class ACEExample<TInput>(
    val input: TInput,
    val label: String,
)

/**
 * Result of ACE optimization.
 *
 * @property playbook The trained playbook containing organized bullet-point advice.
 * @property config An [OptimizationConfig] with the playbook context injected as instruction
 *  override for the target optimizable node.
 * @property targetNodeName The name of the optimizable node that receives the playbook.
 * @property iterations Number of training examples processed.
 */
public data class ACEOptimizationResult(
    val playbook: ACEPlaybook,
    val config: OptimizationConfig,
    val targetNodeName: String,
    val iterations: Int,
)

/**
 * ACE (Automatic Curriculum Expert) optimizer.
 *
 * Builds a playbook of bullet-point advice through a 3-stage pipeline:
 * 1. **Generator**: Runs the agent on training examples with trajectory tracing
 * 2. **Reflector**: Analyzes trajectories via LLM to extract insights
 * 3. **Curator**: Generates delta updates to the playbook based on insights
 *
 * The playbook is injected into the agent's optimizable prompt node as an instruction
 * override via [OptimizationConfig].
 *
 * @property metricThreshold Minimum metric score to consider a run successful.
 */
public class ACE(
    public val metricThreshold: Double = 1.0,
) {
    /**
     * Optimizes the agent by building a playbook from training trajectories.
     *
     * @param TInput The agent's input type.
     * @param TOutput The agent's output type.
     * @param promptExecutor The executor for LLM calls (used for agent execution and reflector/curator).
     * @param agentConfig The agent configuration.
     * @param strategy The agent strategy. Must contain at least one [OptimizableNode].
     * @param toolRegistry Tools available to the agent.
     * @param trainset Training examples with ground truth labels.
     * @param metric Evaluation function: (label, agentOutput) -> score.
     * @param reflectorModel LLM model for the reflector stage.
     * @param curatorModel LLM model for the curator stage.
     * @param initialPlaybook Optional pre-existing playbook to continue training from.
     * @param targetNodeName Name of the optimizable node to inject the playbook into.
     *  If null, it uses the first optimizable node found.
     * @return [ACEOptimizationResult] containing the trained playbook and optimization config.
     */
    @OptIn(InternalAgentsApi::class)
    public suspend fun <TInput, TOutput> optimize(
        promptExecutor: PromptExecutor,
        agentConfig: AIAgentConfig,
        strategy: AIAgentGraphStrategy<TInput, TOutput>,
        toolRegistry: ToolRegistry = ToolRegistry.EMPTY,
        trainset: List<ACEExample<TInput>>,
        metric: (label: String, output: TOutput) -> Double,
        reflectorModel: LLModel,
        curatorModel: LLModel,
        initialPlaybook: ACEPlaybook = ACEPlaybook(),
        targetNodeName: String? = null,
    ): ACEOptimizationResult {
        require(trainset.isNotEmpty()) { "trainset is required for ACE" }

        val optimizableNodes = strategy.findOptimizableNodes()
        require(optimizableNodes.isNotEmpty()) { "Strategy must contain at least one OptimizableNode for ACE" }

        val targetNode = if (targetNodeName != null) {
            optimizableNodes.find { it.name == targetNodeName }
                ?: error("OptimizableNode '$targetNodeName' not found in strategy")
        } else {
            optimizableNodes.first()
        }

        for ((index, example) in trainset.withIndex()) {
            try {
                logger.info { "Processing item #${index + 1} / ${trainset.size}..." }

                // Generator stage: run agent with trajectory collection and playbook injection
                val playbookConfig = buildPlaybookConfig(targetNode, initialPlaybook)

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

                val output = withContext(playbookConfig) { tracingAgent.run(example.input) }

                val score = metric(example.label, output)
                val isSuccess = score >= metricThreshold

                // Reflector stage: extract insights from trajectory
                logger.debug { "Extracting insights from trajectory..." }
                val insights = extractInsights(
                    label = example.label,
                    isSuccess = isSuccess,
                    trajectory = trajectory,
                    usedPlaybook = initialPlaybook,
                    promptExecutor = promptExecutor,
                    model = reflectorModel,
                )
                if (insights == null) {
                    logger.warn { "Failed to extract insights from trajectory. Skipping item." }
                    continue
                }
                initialPlaybook.updateCounters(insights.flatMap { it.bulletTags })

                // Curator stage: generate playbook delta updates
                logger.debug { "Generating delta updates from insights..." }
                val rawDeltaUpdates = generateDeltaUpdates(
                    insights = insights,
                    playbook = initialPlaybook,
                    promptExecutor = promptExecutor,
                    model = curatorModel,
                )
                if (rawDeltaUpdates == null) {
                    logger.warn { "Failed to generate delta updates from insights. Skipping item." }
                    continue
                }
                for (rawDelta in rawDeltaUpdates) {
                    initialPlaybook.applyDelta(rawDelta.toDeltaUpdate())
                }
            } catch (e: Exception) {
                logger.error(e) { "Failed to process item; skipping it. Reason:\n${e.stackTraceToString()}" }
            }
        }

        logger.info { "ACE optimization completed." }

        val finalConfig = buildPlaybookConfig(targetNode, initialPlaybook)
        return ACEOptimizationResult(
            playbook = initialPlaybook,
            config = finalConfig,
            targetNodeName = targetNode.name,
            iterations = trainset.size,
        )
    }

    private fun buildPlaybookConfig(
        targetNode: OptimizableNode<*, *>,
        playbook: ACEPlaybook,
    ): OptimizationConfig {
        val playbookContext = ACEPrompts.constructPlaybookContext(playbook)
        val fullInstruction = if (playbookContext.isNotBlank()) {
            targetNode.instruction + "\n\n" + playbookContext
        } else {
            targetNode.instruction
        }
        return OptimizationConfig(
            instructions = mapOf(targetNode.name to fullInstruction),
        )
    }

    private suspend fun extractInsights(
        label: String,
        isSuccess: Boolean,
        trajectory: CollectedTrajectory,
        usedPlaybook: ACEPlaybook,
        promptExecutor: PromptExecutor,
        model: LLModel,
    ): List<TrajectoryInsight>? {
        val fullPrompt = trajectory.getLatestFullPrompt()
        if (fullPrompt == null) {
            logger.warn { "No trajectory captured (no LLM calls made). Skipping insights extraction." }
            return null
        }
        val prompt = ACEPrompts.reflectorPrompt(
            groundTruthLabel = label,
            usedPlaybook = usedPlaybook,
            isSuccess = isSuccess,
            trajectoryText = fullPrompt.prettyPrint(),
        )
        return promptExecutor.executeStructured<ReflectorResponse>(
            prompt = prompt,
            model = model,
        ).onFailure {
            logger.warn { "Failed to get reflector response from trajectory, cause:\n${it.stackTraceToString()}" }
        }.getOrNull()?.data?.insights
    }

    private suspend fun generateDeltaUpdates(
        insights: List<TrajectoryInsight>,
        playbook: ACEPlaybook,
        promptExecutor: PromptExecutor,
        model: LLModel,
    ): List<RawDeltaUpdate>? {
        val prompt = ACEPrompts.curatorPrompt(insights, playbook)
        return promptExecutor.executeStructured<CuratorResponse>(
            prompt = prompt,
            model = model,
        ).onFailure {
            logger.warn { "Failed to get curator response from insights, cause:\n${it.stackTraceToString()}" }
        }.getOrNull()?.data?.operations
    }
}
