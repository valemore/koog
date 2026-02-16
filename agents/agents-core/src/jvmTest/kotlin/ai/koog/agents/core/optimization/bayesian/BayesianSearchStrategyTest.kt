package ai.koog.agents.core.optimization.bayesian

import ai.koog.agents.core.optimization.core.Demonstration
import ai.koog.agents.core.optimization.core.OptimizationConfig
import ai.koog.agents.core.optimization.optimizers.mipro.bayesian.TPEConfig
import ai.koog.agents.core.optimization.optimizers.mipro.bayesian.bayesianSearch
import kotlinx.coroutines.runBlocking
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class BayesianSearchStrategyTest {

    @Test
    fun testFindsOptimalConfig() = runBlocking {
        val moduleNames = listOf("thinking", "answer")
        val instructionCandidates = mapOf(
            "thinking" to listOf("Think step by step", "Reason carefully", "Analyze deeply"),
            "answer" to listOf("Give a concise answer", "Be detailed", "Summarize"),
        )
        val demoCandidates = mapOf(
            "thinking" to listOf(
                listOf(Demonstration("q1", "t1", false)),
                listOf(Demonstration("q2", "t2", false)),
            ),
            "answer" to listOf(
                listOf(Demonstration("t1", "a1", false)),
                listOf(Demonstration("t2", "a2", false)),
            ),
        )
        val defaultInstructions = mapOf(
            "thinking" to "Think about it",
            "answer" to "Answer the question",
        )

        // Optimal config: instruction index 0, demo index 1 for both modules
        val result = bayesianSearch(
            moduleNames = moduleNames,
            instructionCandidates = instructionCandidates,
            demoCandidates = demoCandidates,
            defaultInstructions = defaultInstructions,
            numTrials = 30,
            tpeConfig = TPEConfig(warmupTrials = 5, numCandidates = 20),
            minibatch = false,
            minibatchFullEvalSteps = 5,
            random = Random(42),
        ) { config, _ ->
            // Score 1.0 only for the specific optimal combination
            val thinkingInstr = config.instructions["thinking"]
            val answerInstr = config.instructions["answer"]
            if (thinkingInstr == "Think step by step" && answerInstr == "Give a concise answer") {
                1.0
            } else {
                0.0
            }
        }

        assertNotNull(result)
        assertTrue(result.score >= 0.0)
        // With 30 trials it should find the optimal config
        assertEquals("Think step by step", result.config.instructions["thinking"])
        assertEquals("Give a concise answer", result.config.instructions["answer"])
    }

    @Test
    fun testReturnsValidOptimizationResult() = runBlocking {
        val moduleNames = listOf("node1")
        val instructionCandidates = mapOf(
            "node1" to listOf("Instruction A", "Instruction B"),
        )
        val defaultInstructions = mapOf("node1" to "Default")

        val result = bayesianSearch(
            moduleNames = moduleNames,
            instructionCandidates = instructionCandidates,
            demoCandidates = null,
            defaultInstructions = defaultInstructions,
            numTrials = 5,
            tpeConfig = TPEConfig(warmupTrials = 3),
            minibatch = false,
            minibatchFullEvalSteps = 5,
            random = Random(42),
        ) { _, _ -> 0.5 }

        assertEquals(5, result.iterations)
        assertEquals("MIPROv2-TPE", result.metadata["optimizer"])
        assertTrue(result.metadata.containsKey("baselineScore"))
        assertTrue(result.metadata.containsKey("numTrials"))
        assertTrue(result.metadata.containsKey("numModules"))
        assertEquals(5, result.metadata["numTrials"])
        assertEquals(1, result.metadata["numModules"])
    }

    @Test
    fun testBaselineEvaluatedFirst() = runBlocking {
        val evaluatedConfigs = mutableListOf<OptimizationConfig>()

        bayesianSearch(
            moduleNames = listOf("node1"),
            instructionCandidates = mapOf("node1" to listOf("A", "B")),
            demoCandidates = null,
            defaultInstructions = mapOf("node1" to "Default"),
            numTrials = 3,
            tpeConfig = TPEConfig(warmupTrials = 2),
            minibatch = false,
            minibatchFullEvalSteps = 5,
            random = Random(42),
        ) { config, _ ->
            evaluatedConfigs.add(config)
            0.5
        }

        // First evaluated config should be the baseline (empty instructions)
        assertTrue(
            evaluatedConfigs.first().instructions.isEmpty(),
            "First evaluation should be the baseline (empty config)"
        )
    }

    @Test
    fun testMinibatchFullEvalCounts() = runBlocking {
        var fullEvalCount = 0
        var minibatchEvalCount = 0

        bayesianSearch(
            moduleNames = listOf("node1"),
            instructionCandidates = mapOf("node1" to listOf("A", "B", "C")),
            demoCandidates = null,
            defaultInstructions = mapOf("node1" to "Default"),
            numTrials = 10,
            tpeConfig = TPEConfig(warmupTrials = 3),
            minibatch = true,
            minibatchFullEvalSteps = 3,
            random = Random(42),
        ) { _, fullEval ->
            if (fullEval) fullEvalCount++ else minibatchEvalCount++
            0.5
        }

        // Full evals: baseline (1) + periodic at trials 3, 6, 9 (3) + final (1) = 5
        assertEquals(5, fullEvalCount, "Expected 5 full evaluations")
        // Minibatch evals: trials 1-10 = 10
        assertEquals(10, minibatchEvalCount, "Expected 10 minibatch evaluations")
    }

    @Test
    fun testNullDemoCandidatesZeroShot() = runBlocking {
        val result = bayesianSearch(
            moduleNames = listOf("node1"),
            instructionCandidates = mapOf("node1" to listOf("A", "B")),
            demoCandidates = null,
            defaultInstructions = mapOf("node1" to "Default"),
            numTrials = 5,
            tpeConfig = TPEConfig(warmupTrials = 3),
            minibatch = false,
            minibatchFullEvalSteps = 5,
            random = Random(42),
        ) { _, _ -> 0.5 }

        // With null demoCandidates, demonstrations should be empty
        assertTrue(
            result.config.demonstrations.isEmpty() ||
                result.config.demonstrations.values.all { it.isEmpty() },
            "Zero-shot mode should produce empty demonstrations"
        )
    }

    @Test
    fun testSingleTrial() = runBlocking {
        val result = bayesianSearch(
            moduleNames = listOf("node1"),
            instructionCandidates = mapOf("node1" to listOf("A")),
            demoCandidates = null,
            defaultInstructions = mapOf("node1" to "Default"),
            numTrials = 1,
            tpeConfig = TPEConfig(warmupTrials = 5),
            minibatch = false,
            minibatchFullEvalSteps = 5,
            random = Random(42),
        ) { _, _ -> 0.75 }

        assertEquals(1, result.iterations)
        assertTrue(result.score >= 0.0)
    }

    @Test
    fun testEmptySearchSpace() = runBlocking {
        // No instruction candidates, no demo candidates → should still work
        // With no candidates, baseline (empty config) always wins since all trials
        // also get default instructions that score the same
        val result = bayesianSearch(
            moduleNames = listOf("node1"),
            instructionCandidates = emptyMap(),
            demoCandidates = null,
            defaultInstructions = mapOf("node1" to "Default instruction"),
            numTrials = 3,
            tpeConfig = TPEConfig(warmupTrials = 1),
            minibatch = false,
            minibatchFullEvalSteps = 5,
            random = Random(42),
        ) { _, _ -> 0.5 }

        assertEquals(3, result.iterations)
        // Result should still have a valid score
        assertTrue(result.score >= 0.0)
    }

    @Test
    fun testBestConfigTracking() = runBlocking {
        var callCount = 0

        val result = bayesianSearch(
            moduleNames = listOf("node1"),
            instructionCandidates = mapOf("node1" to listOf("A", "B", "C")),
            demoCandidates = null,
            defaultInstructions = mapOf("node1" to "Default"),
            numTrials = 10,
            tpeConfig = TPEConfig(warmupTrials = 3),
            minibatch = false,
            minibatchFullEvalSteps = 5,
            random = Random(42),
        ) { config, _ ->
            // Only instruction "B" scores well
            if (config.instructions["node1"] == "B") 0.9 else 0.1
        }

        // Should have found "B" as best
        assertEquals("B", result.config.instructions["node1"])
        assertTrue(result.score > 0.0)
    }
}
