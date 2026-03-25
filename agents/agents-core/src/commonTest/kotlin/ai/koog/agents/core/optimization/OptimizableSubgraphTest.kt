package ai.koog.agents.core.optimization

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.ToolCalls
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.optimization.core.Demonstration
import ai.koog.agents.core.optimization.core.DemonstrationFormat
import ai.koog.agents.core.optimization.core.FewShotPromptType
import ai.koog.agents.core.optimization.core.OptimizationArtifact
import ai.koog.agents.core.optimization.core.optimizableSubgraphWithTask
import ai.koog.agents.core.optimization.features.CollectedSubgraphTraces
import ai.koog.agents.core.optimization.features.SubgraphTraceCollectionFeature
import ai.koog.agents.core.optimization.features.collectSubgraphTraces
import ai.koog.agents.core.optimization.features.installOptimization
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.SubgraphWithTaskUtils
import ai.koog.agents.features.eventHandler.feature.EventHandler
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.message.Message
import ai.koog.utils.io.use
import kotlinx.coroutines.test.runTest
import kotlin.js.JsName
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class OptimizableSubgraphTest {

    private val model = OpenAIModels.Chat.GPT4o

    // The finish tool used internally by optimizableSubgraphWithTask (identityTool<String>)
    private val finishTool = SubgraphWithTaskUtils.finishTool<String>()

    private fun createMockExecutor(): PromptExecutor = getMockExecutor {
        mockLLMToolCall(finishTool, "done") onCondition { true }
    }

    private fun createAgent(
        freshHistory: Boolean = true,
        optimizableInstruction: String = "Default instruction.",
        config: OptimizationArtifact? = null,
        capturedPrompts: MutableList<Prompt>? = null,
        fewShotPromptType: FewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
        demonstrationFormat: DemonstrationFormat = DemonstrationFormat.COMPACT,
        executor: PromptExecutor = createMockExecutor(),
        collectTraces: Boolean = false,
    ): AIAgent<String, String> {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = optimizableInstruction,
                freshHistory = freshHistory,
                fewShotPromptType = fewShotPromptType,
                demonstrationFormat = demonstrationFormat,
                runMode = ToolCalls.SEQUENTIAL,
            ) { instruction, input -> "$instruction\nInput: $input" }

            nodeStart then classify then nodeFinish
        }

        return AIAgent(
            promptExecutor = executor,
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("test") {
                    system("Parent system prompt.")
                    user("Prior conversation.")
                    assistant("Prior response.")
                },
                model = model,
                maxAgentIterations = 20,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                if (config != null) {
                    installOptimization { artifact = config }
                }
                if (capturedPrompts != null) {
                    install(EventHandler) {
                        onLLMCallStarting { capturedPrompts += it.prompt }
                    }
                }
                if (collectTraces) {
                    collectSubgraphTraces { }
                }
            },
        )
    }

    @Test
    @JsName("testUsesDefaultInstructionWhenNoConfigInstalled")
    fun testUsesDefaultInstructionWhenNoConfigInstalled() = runTest {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            optimizableInstruction = "Classify the sentiment.",
            capturedPrompts = prompts,
        ).use { it.run("hello") }

        val firstPrompt = prompts.first()
        assertTrue(
            firstPrompt.messages.any { it.content.contains("Classify the sentiment.") },
            "Should use default instruction when no config installed"
        )
    }

    @Test
    @JsName("testUsesDefaultInstructionWhenConfigHasNoEntryForThisSubgraph")
    fun testUsesDefaultInstructionWhenConfigHasNoEntryForThisSubgraph() = runTest {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            optimizableInstruction = "Default instruction.",
            config = OptimizationArtifact(
                subgraphInstructions = mapOf("other-subgraph" to "Should not appear"),
            ),
            capturedPrompts = prompts,
        ).use { it.run("hello") }

        val firstPrompt = prompts.first()
        assertTrue(firstPrompt.messages.any { it.content.contains("Default instruction.") })
        assertTrue(firstPrompt.messages.none { it.content.contains("Should not appear") })
    }

    @Test
    @JsName("testUsesOptimizedInstructionFromConfig")
    fun testUsesOptimizedInstructionFromConfig() = runTest {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            optimizableInstruction = "Default instruction.",
            config = OptimizationArtifact(
                subgraphInstructions = mapOf("classify" to "Optimized: classify with care."),
            ),
            capturedPrompts = prompts,
        ).use { it.run("hello") }

        val firstPrompt = prompts.first()
        assertTrue(
            firstPrompt.messages.any { it.content.contains("Optimized: classify with care.") },
            "Should use optimized instruction from config"
        )
        assertTrue(
            firstPrompt.messages.none { it.content.contains("Default instruction.") },
            "Default instruction should not appear when config overrides it"
        )
    }

    @Test
    @JsName("testFreshHistoryStartsWithEmptyPromptAndSystemMessage")
    fun testFreshHistoryStartsWithEmptyPromptAndSystemMessage() = runTest {
        val prompts = mutableListOf<Prompt>()

        createAgent(freshHistory = true, capturedPrompts = prompts).use { it.run("test input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages
        val systemMessages = messages.filterIsInstance<Message.System>()
        assertEquals(1, systemMessages.size, "Expected one system message from defineTask")
        assertTrue(messages.none { it.content.contains("Parent system prompt") })
        assertTrue(messages.none { it.content.contains("Prior conversation") })
    }

    @Test
    @JsName("testNonFreshHistoryInheritsParentPromptAndUsesUserMessage")
    fun testNonFreshHistoryInheritsParentPromptAndUsesUserMessage() = runTest {
        val prompts = mutableListOf<Prompt>()

        createAgent(freshHistory = false, capturedPrompts = prompts).use { it.run("test input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages
        assertTrue(messages.filterIsInstance<Message.System>().any {
            it.content.contains("Parent system prompt")
        })
        assertTrue(messages.filterIsInstance<Message.User>().any {
            it.content.contains("Default instruction.")
        })
    }

    @Test
    @JsName("testDemosInjectedAsMessageHistoryAfterTaskDescription")
    fun testDemosInjectedAsMessageHistoryAfterTaskDescription() = runTest {
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(
            Demonstration("example input", "example output"),
            Demonstration("another input", "another output"),
        )

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages

        // System message (from defineTask) should come first
        assertTrue(messages.first() is Message.System)

        // Demo user/assistant pairs should be present
        assertTrue(messages.filterIsInstance<Message.User>().any { it.content == "example input" })
        assertTrue(messages.filterIsInstance<Message.Assistant>().any { it.content == "example output" })
        assertTrue(messages.filterIsInstance<Message.User>().any { it.content == "another input" })
        assertTrue(messages.filterIsInstance<Message.Assistant>().any { it.content == "another output" })
    }

    @Test
    @JsName("testDemosInjectedAsStringProducesSingleUserMessage")
    fun testDemosInjectedAsStringProducesSingleUserMessage() = runTest {
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(Demonstration("example input", "example output"))

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_STRING,
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        val userMessages = firstPrompt.messages.filterIsInstance<Message.User>()
        assertTrue(
            userMessages.any {
                it.content.contains("Input: example input") && it.content.contains("Output: example output")
            },
            "AS_STRING should produce a single user message with rendered demos"
        )
    }

    @Test
    @JsName("testNoDemosInjectedWhenConfigHasNoDemosForSubgraph")
    fun testNoDemosInjectedWhenConfigHasNoDemosForSubgraph() = runTest {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(
                subgraphDemonstrations = mapOf("other" to listOf(Demonstration("x", "y"))),
            ),
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        assertTrue(
            firstPrompt.messages.none { it.content.contains("x") },
            "Demos for other subgraph should not leak into this subgraph"
        )
    }

    @Test
    @JsName("testDemosInjectedInNonFreshHistoryMode")
    fun testDemosInjectedInNonFreshHistoryMode() = runTest {
        val prompts = mutableListOf<Prompt>()
        val demos = listOf(Demonstration("demo-in", "demo-out"))

        createAgent(
            freshHistory = false,
            config = OptimizationArtifact(subgraphDemonstrations = mapOf("classify" to demos)),
            fewShotPromptType = FewShotPromptType.AS_MESSAGE_HISTORY,
            capturedPrompts = prompts,
        ).use { it.run("real input") }

        val firstPrompt = prompts.first()
        val messages = firstPrompt.messages

        assertTrue(messages.filterIsInstance<Message.System>().any {
            it.content.contains("Parent system prompt")
        }, "Parent system prompt should be preserved")
        assertTrue(messages.filterIsInstance<Message.User>().any {
            it.content == "demo-in"
        }, "Demo should be injected")
        assertTrue(messages.filterIsInstance<Message.Assistant>().any {
            it.content == "demo-out"
        }, "Demo should be injected")
    }

    @Test
    @JsName("testSubgraphNameDerivedFromPropertyName")
    fun testSubgraphNameDerivedFromPropertyName() = runTest {
        val prompts = mutableListOf<Prompt>()

        createAgent(
            freshHistory = true,
            config = OptimizationArtifact(
                subgraphInstructions = mapOf("classify" to "Name-resolved instruction"),
            ),
            capturedPrompts = prompts,
        ).use { it.run("test") }

        assertTrue(
            prompts.first().messages.any { it.content.contains("Name-resolved instruction") },
            "Subgraph should resolve its name from the property name for config lookup"
        )
    }

    @Test
    @JsName("testExplicitNameOverridesPropertyName")
    fun testExplicitNameOverridesPropertyName() = runTest {
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val myProperty by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Default.",
                name = "custom-name",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then myProperty then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 20),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf("custom-name" to "Custom instruction"),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        assertTrue(
            prompts.first().messages.any { it.content.contains("Custom instruction") },
            "Should use explicit name 'custom-name' for config lookup, not property name 'myProperty'"
        )
    }

    @Test
    @JsName("testMultipleSubgraphsEachGetOwnInstruction")
    fun testMultipleSubgraphsEachGetOwnInstruction() = runTest {
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First default.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second default.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 40),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf(
                            "first" to "Optimized first",
                            "second" to "Optimized second",
                        ),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        assertTrue(prompts.size >= 2, "Expected at least two LLM calls")
        assertTrue(prompts[0].messages.any { it.content.contains("Optimized first") })
        assertTrue(prompts[0].messages.none { it.content.contains("Optimized second") })
        assertTrue(prompts[1].messages.any { it.content.contains("Optimized second") })
        assertTrue(prompts[1].messages.none { it.content.contains("Optimized first") })
    }

    @Test
    @JsName("testDemosForOneSubgraphDoNotLeakToAnother")
    fun testDemosForOneSubgraphDoNotLeakToAnother() = runTest {
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 40),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installOptimization {
                    artifact = OptimizationArtifact(
                        subgraphDemonstrations = mapOf(
                            "first" to listOf(Demonstration("first-demo-in", "first-demo-out")),
                        ),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        assertTrue(prompts[0].messages.any { it.content.contains("first-demo-in") },
            "First subgraph should see its demos")
        assertTrue(prompts[1].messages.none { it.content.contains("first-demo-in") },
            "Second subgraph should not see first subgraph's demos")
    }

    @Test
    @JsName("testDuplicateNamesCauseConfigCollision")
    fun testDuplicateNamesCauseConfigCollision() = runTest {
        // Documents the known limitation: two subgraphs with the same name share config entries.
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First default.",
                name = "task",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second default.",
                name = "task",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = 40),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf("task" to "Shared instruction"),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
            },
        ).use { it.run("input") }

        // Both subgraphs get the same instruction — this is the collision (known limitation).
        assertTrue(prompts[0].messages.any { it.content.contains("Shared instruction") })
        assertTrue(prompts[1].messages.any { it.content.contains("Shared instruction") },
            "Duplicate names cause both subgraphs to share the same config entry (known limitation)")
    }

    /** Helper to run an agent with trace collection and return the collected traces. */
    private suspend fun runWithTraceCollection(
        strategy: ai.koog.agents.core.agent.entity.AIAgentGraphStrategy<String, String>,
        input: String,
        maxIterations: Int = 40,
    ): CollectedSubgraphTraces {
        val agent = AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(prompt = prompt("t") { }, model = model, maxAgentIterations = maxIterations),
            toolRegistry = ToolRegistry { },
            installFeatures = { collectSubgraphTraces { } },
        )
        val session = agent.createSession()
        val traces = session.pipeline()?.feature(CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature)
            ?: error("Trace collection feature should be installed")
        session.run(input)
        return traces
    }

    @Test
    @JsName("testTraceCollectionCapturesSubgraphInputOutput")
    fun testTraceCollectionCapturesSubgraphInputOutput() = runTest {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "trace-input")
        val classifyTraces = traces.getTraces("classify")
        assertEquals(1, classifyTraces.size, "Should have collected one trace for 'classify'")
        assertEquals("trace-input", classifyTraces.first().input)
    }

    @Test
    @JsName("testTraceCollectionCapturesIntermediateMessages")
    fun testTraceCollectionCapturesIntermediateMessages() = runTest {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val demo = traces.getTraces("classify").first()
        assertNotNull(demo.intermediateMessages,
            "Intermediate messages should be captured by afterFinishToolCall hook")
        assertTrue(demo.intermediateMessages.isNotEmpty(),
            "Intermediate messages should contain the subgraph's conversation")
    }

    @Test
    @JsName("testTraceCollectionCapturesWholeAgentTrajectory")
    fun testTraceCollectionCapturesWholeAgentTrajectory() = runTest {
        val strategy = strategy<String, String>("test-strategy") {
            val classify by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Classify.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then classify then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val fullPrompt = traces.getLatestFullPrompt()
        assertNotNull(fullPrompt, "Whole-agent trajectory should be captured via interceptLLMCallCompleted")
        assertTrue(fullPrompt.messages.isNotEmpty(), "Trajectory should contain messages")
    }

    @Test
    @JsName("testTracesPerSubgraphAreIsolated")
    fun testTracesPerSubgraphAreIsolated() = runTest {
        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val allTraces = traces.getAllTraces()

        assertTrue("first" in allTraces, "Should have traces for 'first' subgraph")
        assertTrue("second" in allTraces, "Should have traces for 'second' subgraph")
        assertEquals(1, allTraces["first"]?.size, "One trace per subgraph execution")
        assertEquals(1, allTraces["second"]?.size, "One trace per subgraph execution")

        // Each subgraph should have its own intermediate messages, not shared
        val firstDemo = allTraces.getValue("first").first()
        val secondDemo = allTraces.getValue("second").first()
        assertNotNull(firstDemo.intermediateMessages)
        assertNotNull(secondDemo.intermediateMessages)

        // The first subgraph's intermediate should contain "First." instruction
        assertTrue(
            firstDemo.intermediateMessages.any { it.content.contains("First.") },
            "First subgraph's intermediate trace should contain its own instruction"
        )
        // The second subgraph's intermediate should contain "Second." instruction
        assertTrue(
            secondDemo.intermediateMessages.any { it.content.contains("Second.") },
            "Second subgraph's intermediate trace should contain its own instruction"
        )
    }

    @Test
    @JsName("testWholeTrajectoryAfterMultipleSubgraphs")
    fun testWholeTrajectoryAfterMultipleSubgraphs() = runTest {
        val strategy = strategy<String, String>("test-strategy") {
            val first by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "First.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            val second by optimizableSubgraphWithTask<String, String>(
                optimizableInstruction = "Second.",
                freshHistory = true,
            ) { instruction, input -> "$instruction\n$input" }

            nodeStart then first then second then nodeFinish
        }

        val traces = runWithTraceCollection(strategy, "input")
        val fullPrompt = traces.getLatestFullPrompt()
        assertNotNull(fullPrompt, "Whole trajectory should be captured")

        // The trajectory is the latest prompt state after the last LLM call.
        // With freshHistory, each subgraph starts fresh, so the latest prompt
        // reflects the second subgraph's conversation (last LLM call).
        assertTrue(fullPrompt.messages.isNotEmpty())
    }

    @Test
    @JsName("testNestedOptimizableSubgraph")
    fun testNestedOptimizableSubgraph() = runTest {
        // Strategy → outer (regular subgraph) → inner (optimizable)
        // Depth 3: strategy > subgraph > optimizableSubgraphWithTask
        val prompts = mutableListOf<Prompt>()

        val strategy = strategy<String, String>("test-strategy") {
            val outer by subgraph<String, String>(name = "outer") {
                val inner by optimizableSubgraphWithTask<String, String>(
                    optimizableInstruction = "Inner default.",
                    freshHistory = true,
                ) { instruction, input -> "$instruction\n$input" }

                nodeStart then inner then nodeFinish
            }

            nodeStart then outer then nodeFinish
        }

        val agent = AIAgent(
            promptExecutor = createMockExecutor(),
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("t") { system("Strategy prompt.") },
                model = model,
                maxAgentIterations = 40,
            ),
            toolRegistry = ToolRegistry { },
            installFeatures = {
                installOptimization {
                    artifact = OptimizationArtifact(
                        subgraphInstructions = mapOf("inner" to "Optimized inner"),
                        subgraphDemonstrations = mapOf(
                            "inner" to listOf(Demonstration("demo-in", "demo-out")),
                        ),
                    )
                }
                install(EventHandler) { onLLMCallStarting { prompts += it.prompt } }
                collectSubgraphTraces { }
            },
        )
        val session = agent.createSession()
        val traces = session.pipeline()?.feature(
            CollectedSubgraphTraces::class, SubgraphTraceCollectionFeature
        )
        session.run("nested-input")

        // Instruction resolved from config
        assertTrue(prompts.isNotEmpty())
        assertTrue(
            prompts.first().messages.any { it.content.contains("Optimized inner") },
            "Nested optimizable subgraph should use config instruction"
        )
        assertTrue(
            prompts.first().messages.none { it.content.contains("Inner default.") },
            "Default instruction should not appear"
        )

        // Demos injected
        assertTrue(
            prompts.first().messages.any { it.content == "demo-in" },
            "Demos should be injected into nested subgraph"
        )
        assertTrue(
            prompts.first().messages.any { it.content == "demo-out" },
            "Demos should be injected into nested subgraph"
        )

        // Traces collected for both outer and inner
        assertNotNull(traces)
        val innerTraces = traces.getTraces("inner")
        assertEquals(1, innerTraces.size, "Should collect trace for nested 'inner'")
        assertEquals("nested-input", innerTraces.first().input)
        assertNotNull(innerTraces.first().intermediateMessages,
            "Nested subgraph should export intermediate messages")

        val outerTraces = traces.getTraces("outer")
        assertEquals(1, outerTraces.size, "Should collect trace for 'outer' wrapper")
    }
}
