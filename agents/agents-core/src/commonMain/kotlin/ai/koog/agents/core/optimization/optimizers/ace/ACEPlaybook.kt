package ai.koog.agents.core.optimization.optimizers.ace

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

private val logger = KotlinLogging.logger {}

/**
 * A single bullet-point advice item within a playbook section.
 *
 * @property content The text content of the advice.
 * @property id Unique identifier (e.g. "strategies-3").
 * @property helpfulCount The number of trajectories where this bullet was tagged helpful.
 * @property harmfulCount The number of trajectories where this bullet was tagged harmful.
 */
@Serializable
public data class PlaybookBulletItem(
    var content: String,
    val id: String,
    var helpfulCount: Int,
    var harmfulCount: Int,
) {
    /** Companion object containing utility functions for working with PlaybookBulletItem objects. */
    public companion object {
        /** Generates a new bullet ID based on the section's short name and existing bullet count. */
        public fun generateNewId(section: PlaybookSection): String = with(section) {
            val bulletNumber = bullets.maxOfOrNull { b ->
                b.id.split("-").last().toInt() + 1
            } ?: 1
            "$shortName-$bulletNumber"
        }
    }

    /**
     * Secondary constructor for creating a PlaybookBulletItem with a generated ID
     * based on the provided section's short name and existing bullet count.
     */
    public constructor(
        content: String,
        section: PlaybookSection,
        helpfulCount: Int,
        harmfulCount: Int,
    ) : this(content, generateNewId(section), helpfulCount, harmfulCount)
}

/**
 * A named section within an ACE playbook containing bullet items.
 *
 * @property name Full display name of the section.
 * @property shortName Short identifier used in bullet IDs (e.g. "strategies").
 * @property bullets Mutable list of bullet items in this section.
 */
@Serializable
public data class PlaybookSection(
    val name: String,
    // TODO: maybe not ask LLM for this?
    val shortName: String,
    val bullets: MutableList<PlaybookBulletItem>,
)

/** A validated playbook update operation. */
@Serializable
public sealed class DeltaUpdate {
    /** Add a new bullet to a section. */
    public data class Add(val content: String, val sectionName: String, val sectionShortName: String) : DeltaUpdate()
    /** Update an existing bullet's content. */
    public data class Update(val id: String, val newContent: String) : DeltaUpdate()
    /** Delete a bullet by ID. */
    public data class Delete(val id: String) : DeltaUpdate()
}

/**
 * A playbook is a collection of bullet-point advice organized in sections.
 *
 * ACE builds and evolves a playbook during training. At inference time, the playbook
 * is injected into the agent's system prompt to guide behavior.
 *
 * The playbook can be serialized/deserialized as JSON for persistence.
 */
public class ACEPlaybook {
    // TODO: add deduplication using embeddings (lazy vs not lazy)
    // TODO: also deduplicate sections

    private val sections = mutableListOf<PlaybookSection>()

    /** Loads playbook sections from a JSON string, appending to existing sections. */
    public fun loadFromJson(jsonString: String) {
        val parsed = json.decodeFromString<List<PlaybookSection>>(jsonString)
        sections.addAll(parsed)
    }

    /** Serializes the playbook to a JSON string. */
    public fun toJsonString(): String = json.encodeToString(sections)

    /** Applies a single delta update (add, update, or delete) to the playbook. */
    public fun applyDelta(delta: DeltaUpdate) {
        when (delta) {
            is DeltaUpdate.Add -> {
                val targetSection = if (sections.any { it.shortName == delta.sectionShortName }) {
                    sections.first { it.shortName == delta.sectionShortName }
                } else {
                    val newSection = PlaybookSection(delta.sectionName, delta.sectionShortName, mutableListOf())
                    sections.add(newSection)
                    newSection
                }
                targetSection.bullets.add(PlaybookBulletItem(delta.content, targetSection, 0, 0))
            }

            is DeltaUpdate.Update -> {
                val bulletItem = sections.flatMap { it.bullets }.find { it.id == delta.id }
                if (bulletItem == null) {
                    logger.warn { "Cannot apply update: bullet with id ${delta.id} not found" }
                    return
                }
                bulletItem.content = delta.newContent
            }

            is DeltaUpdate.Delete -> {
                val bulletItem = sections.flatMap { it.bullets }.find { it.id == delta.id }
                var deleteCount = 0
                for (section in sections) {
                    val deleted = section.bullets.remove(bulletItem)
                    if (deleted) deleteCount++
                }
                if (deleteCount != 1) logger.warn { "Delta-delete: removed $deleteCount bullets instead of 1" }
            }
        }
    }

    /** Increments helpful/harmful counters on bullets based on reflector tags. */
    public fun updateCounters(bulletTags: List<BulletTag>) {
        for ((bulletId, tag) in bulletTags) {
            val bulletItem = sections.flatMap { it.bullets }.find { it.id == bulletId }
            if (bulletItem == null) {
                logger.warn { "Cannot update counter: bullet with id $bulletId not found" }
                continue
            }
            when (tag) {
                "helpful" -> bulletItem.helpfulCount++
                "harmful" -> bulletItem.harmfulCount++
                "neutral" -> {}
                else -> logger.warn { "Unknown tag $tag for bullet $bulletId" }
            }
        }
    }

    /** Renders the playbook as human-readable text for injection into agent prompts. */
    public fun toPromptRepresentation(): String = buildString {
        appendLine(" ======= PLAYBOOK BEGIN =======\n")
        for (section in sections) {
            appendLine(" === SECTION ${section.name} === \n")
            for (bullet in section.bullets) {
                appendLine("[${bullet.id}]")
                appendLine(bullet.content)
                appendLine()
            }
        }
        appendLine("======= PLAYBOOK END =======")
    }

    /** Serializes the playbook to JSON using the provided [Json] instance. */
    public fun toJson(json: Json): String = json.encodeToString(sections)

    private companion object {
        val json = Json { ignoreUnknownKeys = true; prettyPrint = true }
    }
}
