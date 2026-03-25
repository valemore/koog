package ai.koog.agents.core.optimization.core

import ai.koog.agents.core.agent.ToolCalls
import ai.koog.agents.core.agent.context.AIAgentGraphContextBase
import ai.koog.agents.core.agent.entity.AIAgentSubgraph
import ai.koog.agents.core.agent.entity.ToolSelectionStrategy
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.dsl.builder.AIAgentBuilderDslMarker
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphBuilderBase
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphDelegate
import ai.koog.agents.core.optimization.features.intermediateMessagesKey
import ai.koog.agents.core.tools.annotations.InternalAgentToolsApi
import ai.koog.agents.ext.agent.identityTool
import ai.koog.agents.ext.agent.setupSubgraphWithTask
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.processor.ResponseProcessor
import kotlin.reflect.KProperty

/**
 * Holds the resolved subgraph name for use inside node lambdas.
 *
 * The name is resolved at property delegation time (graph construction) and read
 * at node execution time. Graph construction always completes before execution starts,
 * so the name is guaranteed to be set when nodes read it.
 */
// TODO: Is this a redundant holder? I believe we can capture names directly in the delegate.
@PublishedApi
internal class SubgraphNameHolder {
    var name: String? = null
}

/**
 * Delegate wrapper for optimizable subgraphs that captures the resolved subgraph name.
 *
 * @param Input The input type for the subgraph.
 * @param Output The output type for the subgraph.
 * @property innerDelegate The underlying subgraph delegate.
 * @property nameHolder Holder populated with the resolved name at delegation time.
 */
public class OptimizableSubgraphDelegate<Input, Output> @PublishedApi internal constructor(
    private val innerDelegate: AIAgentSubgraphDelegate<Input, Output>,
    @PublishedApi internal val nameHolder: SubgraphNameHolder,
) {
    /**
     * Property delegation operator. Resolves the subgraph name and makes it available
     * to node lambdas via the [nameHolder].
     */
    public operator fun getValue(thisRef: Any?, property: KProperty<*>): AIAgentSubgraph<Input, Output> {
        val subgraph = innerDelegate.getValue(thisRef, property)
        nameHolder.name = subgraph.name
        return subgraph
    }
}

/**
 * Creates an optimizable subgraph that performs a single task with a tunable instruction and
 * bootstrappable few-shot demonstrations.
 *
 * This is the optimization-aware counterpart of `subgraphWithTask`. Each optimizable subgraph
 * acts as a "module" (in DSPy terms) whose instruction and demonstrations can be tuned by
 * an optimizer without modifying the graph.
 *
 * The three differences from `subgraphWithTask`:
 * 1. **Instruction resolution**: the effective instruction is read from [OptimizationArtifact] in
 *    storage (falling back to [optimizableInstruction]), then passed to [defineTask].
 * 2. **Demo injection**: demonstrations from [OptimizationArtifact] are injected into the prompt
 *    after the task description, before the LLM request.
 * 3. **Trace export**: intermediate messages are saved to storage before the prompt is discarded,
 *    enabling [SubgraphTraceCollectionFeature][ai.koog.agents.core.optimization.features.SubgraphTraceCollectionFeature]
 *    to capture full execution traces.
 *
 * If [OptimizationFeature][ai.koog.agents.core.optimization.features.OptimizationFeature]
 * is not installed, the subgraph uses [optimizableInstruction] and empty demonstrations.
 *
 * **Important**: subgraph names must be globally unique within a strategy for optimization to
 * work correctly. The [OptimizationArtifact] uses the subgraph name as a lookup key — duplicate
 * names will cause incorrect instruction/demo assignment. If [name] is not provided, the
 * property name is used (same convention as regular nodes and subgraphs in koog).
 *
 * TODO: enforce global uniqueness of optimizable subgraph names within a strategy at
 *  construction time. Currently, duplicate names are not detected and silently cause
 *  incorrect optimization behavior.
 *
 * @param Input The input type for the subgraph.
 * @param Output The output type for the subgraph.
 * @param optimizableInstruction Default instruction, overridable by [OptimizationArtifact].
 * @param name Optional subgraph name. If null, derived from the delegated property name.
 *   Used as the key for [OptimizationArtifact] lookup.
 * @param toolSelectionStrategy Strategy for selecting available tools.
 * @param llmModel Optional LLM model override.
 * @param llmParams Optional LLM parameters override.
 * @param runMode Tool execution mode (sequential, parallel, single-run).
 * @param assistantResponseRepeatMax Max retries when model doesn't call tools.
 * @param responseProcessor Optional post-processing of LLM responses.
 * @param freshHistory When true, the subgraph starts with an empty conversation history.
 * @param fewShotPromptType How demos are inserted. Null inherits from [PromptInsertionDefaults] in storage.
 * @param demonstrationFormat Detail level for demos. Null inherits from [PromptInsertionDefaults] in storage.
 * @param defineTask Lambda that composes the task description from the resolved instruction and input.
 * @return A delegate for use with Kotlin property delegation (`by`).
 */
@OptIn(InternalAgentToolsApi::class, InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified Input, reified Output> AIAgentSubgraphBuilderBase<*, *>.optimizableSubgraphWithTask(
    optimizableInstruction: String,
    name: String? = null,
    toolSelectionStrategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    llmModel: LLModel? = null,
    llmParams: LLMParams? = null,
    runMode: ToolCalls = ToolCalls.SEQUENTIAL,
    assistantResponseRepeatMax: Int? = null,
    responseProcessor: ResponseProcessor? = null,
    freshHistory: Boolean = false,
    fewShotPromptType: FewShotPromptType? = null,
    demonstrationFormat: DemonstrationFormat? = null,
    noinline defineTask: suspend AIAgentGraphContextBase.(instruction: String, input: Input) -> String,
): OptimizableSubgraphDelegate<Input, Output> {
    val finishTool = identityTool<Output>()
    val nameHolder = SubgraphNameHolder()

    val innerDelegate = subgraph<Input, Output>(
        name = name,
        toolSelectionStrategy = toolSelectionStrategy,
        llmModel = llmModel,
        llmParams = llmParams,
        responseProcessor = responseProcessor,
        freshHistory = freshHistory,
    ) {
        setupSubgraphWithTask<Input, Output, Output>(
            finishTool = finishTool,
            runMode = runMode,
            assistantResponseRepeatMax = assistantResponseRepeatMax,
            freshHistory = freshHistory,

            // Resolve instruction from OptimizationArtifact, pass to user's defineTask
            defineTask = defineTask@{ input ->
                val subgraphName = nameHolder.name
                    ?: error("Optimizable subgraph name was not resolved. This is a framework bug.")
                val artifact = storage.get(OptimizationArtifact.STORAGE_KEY)
                val effectiveInstruction = artifact?.getInstruction(subgraphName)
                    ?: optimizableInstruction

                defineTask(effectiveInstruction, input)
            },

            // Inject demonstrations after task description, before LLM request
            beforeLLMRequest = beforeLLMRequest@{
                val subgraphName = nameHolder.name ?: return@beforeLLMRequest
                val artifact = storage.get(OptimizationArtifact.STORAGE_KEY)
                val demos = artifact?.getDemonstrations(subgraphName).orEmpty()
                if (demos.isEmpty()) return@beforeLLMRequest

                val defaults = storage.get(PromptInsertionDefaults.STORAGE_KEY)
                val effectivePromptType = fewShotPromptType
                    ?: defaults?.fewShotPromptType
                    ?: FewShotPromptType.AS_MESSAGE_HISTORY
                val effectiveFormat = demonstrationFormat
                    ?: defaults?.demonstrationFormat
                    ?: DemonstrationFormat.COMPACT

                llm.writeSession {
                    when (effectivePromptType) {
                        FewShotPromptType.AS_STRING -> {
                            val rendered = DemonstrationRenderer.renderAsString(demos, effectiveFormat)
                            if (rendered != null) {
                                appendPrompt { user(rendered) }
                            }
                        }

                        FewShotPromptType.AS_MESSAGE_HISTORY -> {
                            val demoMessages = DemonstrationRenderer.renderAsMessages(demos, effectiveFormat)
                            if (demoMessages.isNotEmpty()) {
                                appendPrompt { messages(demoMessages) }
                            }
                        }
                    }
                }
            },

            // Export intermediate messages for trace collection
            afterFinishToolCall = afterFinishToolCall@{
                val subgraphName = nameHolder.name ?: return@afterFinishToolCall
                val messages = llm.readSession { prompt.messages }
                storage.set(intermediateMessagesKey(subgraphName), messages)
            },
        )
    }

    return OptimizableSubgraphDelegate(innerDelegate, nameHolder)
}
