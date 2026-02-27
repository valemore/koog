package ai.koog.agents.core.optimization.optimizers.reasoningbank

import kotlinx.serialization.Serializable

/**
 * A single memory extracted from an agent trajectory.
 *
 * @property query The original query that produced this memory.
 * @property title Short title of the memory item.
 * @property description One-sentence summary.
 * @property content Detailed insight (1-5 sentences).
 * @property type Whether this memory came from a successful or failed trajectory.
 * @property id Unique identifier for this memory.
 */
@Serializable
public data class MemoryItem(
    val query: String,
    val title: String,
    val description: String,
    val content: String,
    val type: MemoryType,
    val id: String,
)

/** Whether a memory item was extracted from a successful or failed trajectory. */
@Serializable
public enum class MemoryType {
    /** Memory extracted from a trajectory that met the success threshold. */
    FROM_SUCCESS,
    /** Memory extracted from a trajectory that failed the success threshold. */
    FROM_FAILURE,
}

/**
 * Interface for computing text embeddings.
 *
 * Implementations provide platform-specific embedding functionality (e.g., Grazie, OpenAI).
 * The embedder converts text into a vector representation for semantic similarity search.
 */
public fun interface TextEmbedder {
    /**
     * Computes the embedding vector for the given text.
     *
     * @param text The text to embed.
     * @return A list of doubles representing the embedding vector.
     */
    public suspend fun embed(text: String): List<Double>
}
