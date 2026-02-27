package ai.koog.agents.core.optimization.optimizers.reasoningbank

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.math.sqrt

private val logger = KotlinLogging.logger {}

@Serializable
internal data class MemoryItemWithEmbedding(
    val item: MemoryItem,
    val embedding: List<Double>,
)

/**
 * A memory bank for storing and retrieving agent experiences.
 *
 * Memories are extracted from agent trajectories during training and retrieved
 * at inference time using cosine similarity between the query embedding and
 * stored memory embeddings.
 *
 * @property embedder The text embedder used for computing similarity.
 */
public class ReasoningBank(
    private val embedder: TextEmbedder,
) {
    private val memories = mutableListOf<MemoryItemWithEmbedding>()

    /**
     * Loads memories from a JSON string.
     */
    public fun loadFromJson(jsonString: String) {
        val parsed = json.decodeFromString<List<MemoryItemWithEmbedding>>(jsonString)
        memories.addAll(parsed)
        logger.info { "Loaded ${parsed.size} memories (total: ${memories.size})" }
    }

    /**
     * Serializes all memories to a JSON string.
     */
    public fun toJsonString(): String = json.encodeToString(memories)

    /**
     * Retrieves the top-k most relevant memories for the given query.
     *
     * @param query The query text to match against.
     * @param k Maximum number of memories to return.
     * @param onlySuccesses If true, only return memories from successful trajectories.
     * @return The most relevant memories, sorted by similarity (descending).
     */
    public suspend fun retrieve(
        query: String,
        k: Int,
        onlySuccesses: Boolean,
    ): List<MemoryItem> {
        val filteredMemories = if (onlySuccesses) {
            memories.filter { it.item.type == MemoryType.FROM_SUCCESS }
        } else {
            memories
        }
        if (filteredMemories.isEmpty()) return emptyList()

        val queryVec = embedder.embed(query)
        return filteredMemories
            .map { mem -> mem to cosineSimilarity(queryVec, mem.embedding) }
            .sortedByDescending { it.second }
            .take(k)
            .map { it.first.item }
    }

    /**
     * Adds a memory item to the bank with its embedding.
     */
    public suspend fun addMemory(item: MemoryItem) {
        val embedding = embedder.embed(item.content)
        memories.add(MemoryItemWithEmbedding(item, embedding))
    }

    private companion object {
        val json = Json { ignoreUnknownKeys = true; prettyPrint = true }

        fun cosineSimilarity(a: List<Double>, b: List<Double>): Double {
            if (a.size != b.size || a.isEmpty()) return 0.0
            var dot = 0.0
            var normA = 0.0
            var normB = 0.0
            for (i in a.indices) {
                dot += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }
            val denom = sqrt(normA) * sqrt(normB)
            return if (denom == 0.0) 0.0 else dot / denom
        }
    }
}
