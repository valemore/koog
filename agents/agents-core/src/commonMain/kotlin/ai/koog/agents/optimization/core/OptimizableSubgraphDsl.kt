package ai.koog.agents.optimization.core

import ai.koog.agents.core.agent.ToolCalls
import ai.koog.agents.core.agent.context.AIAgentGraphContextBase
import ai.koog.agents.core.agent.entity.AIAgentSubgraph
import ai.koog.agents.core.agent.entity.ToolSelectionStrategy
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.dsl.builder.AIAgentBuilderDslMarker
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphBuilderBase
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphDelegate
import ai.koog.agents.optimization.features.inheritedMessagesKey
import ai.koog.agents.optimization.features.intermediateMessagesKey
import ai.koog.agents.core.tools.annotations.InternalAgentToolsApi
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.ext.agent.identityTool
import ai.koog.agents.ext.agent.setupSubgraphWithTask
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.processor.ResponseProcessor
import kotlin.reflect.KProperty

/** Wrapper to store nullable or platform-typed Input in a storage key that requires `Any`. */
@PublishedApi
internal class InputHolder<Input>(val value: Input)

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
 *    enabling [SubgraphTraceCollectionFeature][ai.koog.agents.optimization.features.SubgraphTraceCollectionFeature]
 *    to capture full execution traces.
 *
 * If [OptimizationFeature][ai.koog.agents.optimization.features.OptimizationFeature]
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
 * @param assistantResponseRepeatMax Max retries when the model doesn't call tools.
 * @param responseProcessor Optional post-processing of LLM responses.
 * @param freshHistory When true, the subgraph starts with an empty conversation history.
 * @param fewShotPromptType How demos are inserted. Null inherits from [PromptInsertionDefaults] in storage.
 * @param demonstrationFormat Detail level for demos. Null inherits from [PromptInsertionDefaults] in storage.
 * @param defineTask Lambda that composes the user query from the resolved instruction and input.
 *   For fresh history, the resolved instruction is also placed as the system message separately,
 *   so demonstrations are sandwiched between the instruction and the query:
 *   `system(instruction) → demos → user(defineTask(instruction, input)) → LLM response`.
 *   The instruction is available in the lambda for convenience — if used, it will appear in both
 *   the system message and the user query (which is fine, it reinforces the instruction).
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

            // Resolve instruction from OptimizationArtifact.
            // For fresh history: returns instruction only (becomes system message);
            //   the user query is appended after demos by beforeLLMRequest.
            // For non-fresh: returns defineTask(instruction, input) (becomes user message);
            //   demos are injected before it by nodeBeforeLLM.
            defineTask = defineTask@{ input ->
                val subgraphName = nameHolder.name
                    ?: error("Optimizable subgraph name was not resolved. This is a framework bug.")

                // Capture the inherited prompt before the subgraph adds anything.
                // Used later to strip the inherited prefix from intermediate messages.
                val inherited = llm.readSession { prompt.messages }
                storage.set(inheritedMessagesKey(subgraphName), inherited)

                val artifact = storage.get(OptimizationArtifact.STORAGE_KEY)
                val effectiveInstruction = artifact?.getInstruction(subgraphName)
                    ?: optimizableInstruction

                // Store input so beforeLLMRequest can append the query after demos.
                // Key includes subgraph name to avoid collisions between subgraphs.
                // TODO: Seek a cleaner way to pass input to the next step, also, need to
                //  re-think the prompt ordering options
                val inputStorageKey = createStorageKey<InputHolder<Input>>("optimizable-subgraph-input-$subgraphName")
                storage.set(inputStorageKey, InputHolder(input))

                if (freshHistory) {
                    // Instruction alone becomes the system message.
                    // The user query (defineTask result) is appended after demos by beforeLLMRequest.
                    effectiveInstruction
                } else {
                    // Non-fresh: combine into a single user message (current behavior).
                    defineTask(effectiveInstruction, input)
                }
            },

            // Inject demonstrations and (for fresh history) the user query.
            //
            // Prompt ordering:
            //   Fresh:     system(instruction) → demos → user(defineTask(instruction, input)) → LLM
            //   Non-fresh: [inherited] → demos → user(defineTask(instruction, input)) → LLM
            beforeLLMRequest = beforeLLMRequest@{
                val subgraphName = nameHolder.name ?: return@beforeLLMRequest
                val artifact = storage.get(OptimizationArtifact.STORAGE_KEY)
                val demos = artifact?.getDemonstrations(subgraphName).orEmpty()

                if (demos.isNotEmpty()) {
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
                                val demoMessages =
                                    DemonstrationRenderer.renderAsMessages(demos, effectiveFormat)
                                if (demoMessages.isNotEmpty()) {
                                    appendPrompt { messages(demoMessages) }
                                }
                            }
                        }
                    }
                }

                // For fresh history: append the user query after demos so it's the last
                // message before the LLM response, not buried in the system message.
                if (freshHistory) {
                    val inputKey = createStorageKey<InputHolder<Input>>("optimizable-subgraph-input-$subgraphName")
                    val input = storage.getValue(inputKey).value
                    val freshArtifact = storage.get(OptimizationArtifact.STORAGE_KEY)
                    val effectiveInstruction = freshArtifact?.getInstruction(subgraphName)
                        ?: optimizableInstruction
                    val queryText = defineTask(effectiveInstruction, input)
                    llm.writeSession {
                        appendPrompt { user(queryText) }
                    }
                }
            },

            // Export intermediate messages for trace collection.
            // Cleans up the captured trace to produce a natural few-shot demonstration:
            //   1. Strip inherited prefix (parent conversation before this subgraph)
            //   2. Drop the leading system message (the instruction — already provided separately)
            //   3. Convert finalize_task_result Tool.Call → plain Assistant message
            //   4. Drop the finalize_task_result Tool.Result (post-answer framework echo)
            // TODO: Message filtering could be cleaner
            afterFinishToolCall = afterFinishToolCall@{
                val subgraphName = nameHolder.name ?: return@afterFinishToolCall
                val allMessages = llm.readSession { prompt.messages }
                val inherited = storage.get(inheritedMessagesKey(subgraphName)).orEmpty()
                val subgraphOnly = DemonstrationRenderer.dropInheritedPrefix(allMessages, inherited)

                val cleaned = subgraphOnly
                    // Drop the leading system message (instruction is already in the system prompt)
                    .dropWhile { it is Message.System }
                    // TODO: For freshHistory = false, the system message is still present in all few-shots.
                    //  We might wanna split those in two as well. Depends on the semantic we want to capture.
                    // Convert finalize_task_result `Tool.Call` to an `Assistant` and drop its Tool.Result
                    // This is needed because we don't want duplicate content like Tool.Call and Tool.Result
                    // to flood the few-shot demonstrations, at the same time, abandoned tool calls w/o results are
                    // not allowed, therefore we remove the Tool.Result and map the Tool.Call to an Assistant.
                    .map { msg ->
                        if (msg is Message.Tool.Call && msg.tool == SubgraphWithTaskUtils.FINALIZE_SUBGRAPH_TOOL_NAME) {
                            Message.Assistant(msg.content, msg.metaInfo)
                        } else {
                            msg
                        }
                    }
                    .filter { msg ->
                        !(msg is Message.Tool.Result && msg.tool == SubgraphWithTaskUtils.FINALIZE_SUBGRAPH_TOOL_NAME)
                    }

                storage.set(intermediateMessagesKey(subgraphName), cleaned)
            },
        )
    }

    return OptimizableSubgraphDelegate(innerDelegate, nameHolder)
}
