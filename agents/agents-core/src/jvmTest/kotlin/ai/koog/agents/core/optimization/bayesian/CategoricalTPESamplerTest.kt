package ai.koog.agents.core.optimization.bayesian

import ai.koog.agents.core.optimization.optimizers.mipro.bayesian.CategoricalDimension
import ai.koog.agents.core.optimization.optimizers.mipro.bayesian.CategoricalTPESampler
import ai.koog.agents.core.optimization.optimizers.mipro.bayesian.Observation
import ai.koog.agents.core.optimization.optimizers.mipro.bayesian.TPEConfig
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CategoricalTPESamplerTest {

    // ---- Warmup tests ----

    @Test
    fun testWarmupReturnsValidIndices() {
        val dims = listOf(
            CategoricalDimension("a", 5),
            CategoricalDimension("b", 3),
        )
        val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 10))
        val random = Random(42)

        repeat(10) {
            val config = sampler.suggest(emptyList(), random)
            assertEquals(2, config.size)
            assertTrue(config["a"]!! in 0 until 5, "a index out of range: ${config["a"]}")
            assertTrue(config["b"]!! in 0 until 3, "b index out of range: ${config["b"]}")
        }
    }

    @Test
    fun testWarmupRespectsThreshold() {
        val dims = listOf(CategoricalDimension("a", 5))
        val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 3))

        assertTrue(sampler.isWarmingUp(0))
        assertTrue(sampler.isWarmingUp(2))
        assertFalse(sampler.isWarmingUp(3))
        assertFalse(sampler.isWarmingUp(100))
    }

    @Test
    fun testDefaultWarmupCalculation() {
        // Default warmup = max(5, numDimensions + 1)
        val twoDims = listOf(
            CategoricalDimension("a", 3),
            CategoricalDimension("b", 3),
        )
        val samplerSmall = CategoricalTPESampler(twoDims, TPEConfig(warmupTrials = null))
        // max(5, 2+1) = 5
        assertTrue(samplerSmall.isWarmingUp(4))
        assertFalse(samplerSmall.isWarmingUp(5))

        val sixDims = (0..5).map { CategoricalDimension("d$it", 3) }
        val samplerLarge = CategoricalTPESampler(sixDims, TPEConfig(warmupTrials = null))
        // max(5, 6+1) = 7
        assertTrue(samplerLarge.isWarmingUp(6))
        assertFalse(samplerLarge.isWarmingUp(7))
    }

    // ---- Splitting tests ----

    @Test
    fun testSplitWithGamma025() {
        val dims = listOf(CategoricalDimension("a", 5))
        val sampler = CategoricalTPESampler(dims, TPEConfig(gamma = 0.25))

        val observations = (0 until 8).map { i ->
            Observation(config = mapOf("a" to i % 5), score = i.toDouble())
        }

        val (good, bad) = sampler.splitObservations(observations)

        // gamma=0.25, 8 obs -> numGood = max(1, min(7, floor(8*0.25))) = max(1, min(7, 2)) = 2
        assertEquals(2, good.size, "Expected 2 good observations")
        assertEquals(6, bad.size, "Expected 6 bad observations")

        // Good should contain the highest-scoring
        assertTrue(good.all { it.score >= 6.0 }, "Good group should contain top scores")
    }

    @Test
    fun testSplitGuaranteesAtLeastOneEach() {
        val dims = listOf(CategoricalDimension("a", 5))
        val sampler = CategoricalTPESampler(dims, TPEConfig(gamma = 0.01)) // Very small gamma

        // With 2 observations: must have at least 1 in each group
        val observations = listOf(
            Observation(mapOf("a" to 0), 1.0),
            Observation(mapOf("a" to 1), 0.0),
        )
        val (good, bad) = sampler.splitObservations(observations)
        assertEquals(1, good.size, "Must have at least 1 good")
        assertEquals(1, bad.size, "Must have at least 1 bad")
    }

    // ---- Distribution tests ----

    @Test
    fun testLaplaceSmoothingGivesNonzeroProbability() {
        val dim = CategoricalDimension("a", 4)
        val dims = listOf(dim)
        val sampler = CategoricalTPESampler(dims, TPEConfig(priorWeight = 1.0))

        // All observations choose category 0
        val group = listOf(
            Observation(mapOf("a" to 0), 1.0),
            Observation(mapOf("a" to 0), 0.9),
            Observation(mapOf("a" to 0), 0.8),
        )

        val dist = sampler.estimateCategoricalDistribution(dim, group)

        assertEquals(4, dist.size)
        // Category 0: (1.0 + 3) / (4.0 + 3) = 4/7
        // Categories 1,2,3: 1.0 / 7 each
        assertTrue(dist[0] > dist[1], "Observed category should have higher probability")
        assertTrue(dist[1] > 0.0, "Unseen categories should have nonzero probability")
        assertTrue(dist[2] > 0.0)
        assertTrue(dist[3] > 0.0)

        // Sum to ~1.0
        val sum = dist.sum()
        assertTrue(sum > 0.999 && sum < 1.001, "Probabilities should sum to 1.0, got $sum")
    }

    @Test
    fun testUniformDistributionWithNoCounts() {
        val dim = CategoricalDimension("a", 3)
        val dims = listOf(dim)
        val sampler = CategoricalTPESampler(dims, TPEConfig(priorWeight = 1.0))

        // Empty group: only Laplace prior → uniform
        val dist = sampler.estimateCategoricalDistribution(dim, emptyList())

        val expected = 1.0 / 3.0
        for (i in 0 until 3) {
            assertTrue(
                dist[i] > expected - 0.001 && dist[i] < expected + 0.001,
                "Expected ~$expected for category $i, got ${dist[i]}"
            )
        }
    }

    // ---- Convergence tests ----

    @Test
    fun testConvergesToBestCategory() {
        val dims = listOf(CategoricalDimension("a", 5))
        val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 5, numCandidates = 50))
        val random = Random(123)

        // Build observations where category 0 always scores 1.0, rest score 0.0
        val observations = (0 until 20).map { i ->
            val cat = i % 5
            Observation(mapOf("a" to cat), if (cat == 0) 1.0 else 0.0)
        }

        // After these observations, sampler should heavily favor category 0
        var countZero = 0
        val numSuggestions = 100
        repeat(numSuggestions) {
            val suggestion = sampler.suggest(observations, random)
            if (suggestion["a"] == 0) countZero++
        }

        assertTrue(
            countZero > 80,
            "Expected >80% of suggestions to be category 0, got $countZero/$numSuggestions"
        )
    }

    @Test
    fun testMultiDimensionConvergence() {
        val dims = listOf(
            CategoricalDimension("d1", 4),
            CategoricalDimension("d2", 3),
        )
        val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 5, numCandidates = 50))
        val random = Random(456)

        // Best config: d1=0, d2=1 -> score 1.0; all others -> 0.0
        val observations = mutableListOf<Observation>()
        for (i in 0 until 4) {
            for (j in 0 until 3) {
                val score = if (i == 0 && j == 1) 1.0 else 0.0
                observations.add(Observation(mapOf("d1" to i, "d2" to j), score))
            }
        }

        var countBest = 0
        val numSuggestions = 100
        repeat(numSuggestions) {
            val suggestion = sampler.suggest(observations, random)
            if (suggestion["d1"] == 0 && suggestion["d2"] == 1) countBest++
        }

        assertTrue(
            countBest > 25,
            "Expected >25% of suggestions to be best config (random baseline ~8%), got $countBest/$numSuggestions"
        )
    }

    // ---- Edge cases ----

    @Test
    fun testSingleCandidatePerDimension() {
        val dims = listOf(
            CategoricalDimension("a", 1),
            CategoricalDimension("b", 1),
        )
        val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 0))
        val random = Random(42)

        val observations = listOf(
            Observation(mapOf("a" to 0, "b" to 0), 0.5),
        )

        val config = sampler.suggest(observations, random)
        assertEquals(0, config["a"])
        assertEquals(0, config["b"])
    }

    @Test
    fun testAllSameScore() {
        val dims = listOf(CategoricalDimension("a", 5))
        val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 3))
        val random = Random(42)

        // All observations have same score — split should still work
        val observations = (0 until 5).map { i ->
            Observation(mapOf("a" to i), 0.5)
        }

        // Should not throw, and should return valid config
        val config = sampler.suggest(observations, random)
        assertTrue(config["a"]!! in 0 until 5)
    }

    @Test
    fun testDeterminismWithSameSeed() {
        val dims = listOf(
            CategoricalDimension("a", 5),
            CategoricalDimension("b", 3),
        )
        val observations = (0 until 10).map { i ->
            Observation(mapOf("a" to i % 5, "b" to i % 3), i * 0.1)
        }

        val results1 = mutableListOf<Map<String, Int>>()
        val results2 = mutableListOf<Map<String, Int>>()

        repeat(20) {
            val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 5))
            results1.add(sampler.suggest(observations, Random(999)))
        }

        repeat(20) {
            val sampler = CategoricalTPESampler(dims, TPEConfig(warmupTrials = 5))
            results2.add(sampler.suggest(observations, Random(999)))
        }

        assertEquals(results1, results2, "Same seed should produce identical results")
    }

    // ---- Validation tests ----

    @Test
    fun testInvalidGamma() {
        assertFailsWith<IllegalArgumentException> {
            TPEConfig(gamma = 0.0)
        }
        assertFailsWith<IllegalArgumentException> {
            TPEConfig(gamma = 1.0)
        }
        assertFailsWith<IllegalArgumentException> {
            TPEConfig(gamma = -0.1)
        }
    }

    @Test
    fun testEmptyDimensions() {
        assertFailsWith<IllegalArgumentException> {
            CategoricalTPESampler(emptyList())
        }
    }

    @Test
    fun testDuplicateDimensionNames() {
        assertFailsWith<IllegalArgumentException> {
            CategoricalTPESampler(
                listOf(
                    CategoricalDimension("a", 3),
                    CategoricalDimension("a", 5),
                )
            )
        }
    }

    @Test
    fun testZeroCategoryDimension() {
        assertFailsWith<IllegalArgumentException> {
            CategoricalDimension("a", 0)
        }
    }

    @Test
    fun testNegativePriorWeight() {
        assertFailsWith<IllegalArgumentException> {
            TPEConfig(priorWeight = -1.0)
        }
    }

    @Test
    fun testZeroNumCandidates() {
        assertFailsWith<IllegalArgumentException> {
            TPEConfig(numCandidates = 0)
        }
    }

    // ---- EI score tests ----

    @Test
    fun testEIScoreFavorsGoodDistribution() {
        val dims = listOf(CategoricalDimension("a", 3))
        val sampler = CategoricalTPESampler(dims, TPEConfig())

        val goodDists = mapOf("a" to doubleArrayOf(0.8, 0.1, 0.1))
        val badDists = mapOf("a" to doubleArrayOf(0.1, 0.5, 0.4))

        val eiCat0 = sampler.computeEIScore(mapOf("a" to 0), goodDists, badDists)
        val eiCat1 = sampler.computeEIScore(mapOf("a" to 1), goodDists, badDists)
        val eiCat2 = sampler.computeEIScore(mapOf("a" to 2), goodDists, badDists)

        assertTrue(eiCat0 > eiCat1, "Category 0 should have higher EI than category 1")
        assertTrue(eiCat0 > eiCat2, "Category 0 should have higher EI than category 2")
    }
}
