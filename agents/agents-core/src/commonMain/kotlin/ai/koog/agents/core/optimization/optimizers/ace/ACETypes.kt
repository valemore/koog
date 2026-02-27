package ai.koog.agents.core.optimization.optimizers.ace

import ai.koog.agents.core.tools.annotations.LLMDescription
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** LLM-structured response from the ACE reflector stage. */
@Serializable
public data class ReflectorResponse(
    /** Extracted insights from the trajectory. */
    val insights: List<TrajectoryInsight>,
)

/** A single insight extracted from an agent trajectory by the reflector. */
@Serializable
@LLMDescription("An insight extracted from an agent trajectory.")
public data class TrajectoryInsight(
    @property:LLMDescription("Your chain of thought / reasoning / thinking process, detailed analysis and calculations")
    val reasoning: String,
    @property:LLMDescription("What specifically went wrong in the reasoning?")
    val errorIdentification: String,
    @property:LLMDescription("Why did this error occur? What concept was misunderstood?")
    val rootCauseAnalysis: String,
    @property:LLMDescription("What should the model have done instead?")
    val correctApproach: String,
    @property:LLMDescription("What strategy, formula, or principle should be remembered to avoid this error?")
    val keyInsight: String,
    @property:LLMDescription("A list of json objects with bullet ID and tag for each bulletpoint used by the trajectory")
    val bulletTags: List<BulletTag>,
)

/** Tag assigned to a playbook bullet indicating whether it was helpful, harmful, or neutral. */
@Serializable
public data class BulletTag(
    /** ID of the playbook bullet item. */
    val id: String,
    @property:LLMDescription("One of: helpful | harmful | neutral")
    val tag: String,
)

/** LLM-structured response from the ACE curator stage containing playbook update operations. */
@Serializable
@LLMDescription("The updates to the playbook that should be applied to the agent to improve its performance on the current task.")
public data class CuratorResponse(
    @property:LLMDescription("Your chain of thought / reasoning / thinking process, detailed analysis and calculations")
    val reasoning: String,
    @property:LLMDescription("A list of updates to be applied to the playbook for future runs")
    val operations: List<RawDeltaUpdate>,
)

/**
 * Raw LLM-produced playbook delta update, deserialized directly from structured output.
 *
 * Use [toDeltaUpdate] to convert to the validated [DeltaUpdate] type.
 */
@Serializable
@SerialName("DeltaUpdate")
@Suppress("unused")
public sealed class RawDeltaUpdate {
    /** The type of update operation. */
    @property:LLMDescription("One of 'add', 'update', or 'delete', based on the update type")
    public abstract val type: String

    /** Converts this raw LLM output to a validated [DeltaUpdate]. */
    public abstract fun toDeltaUpdate(): DeltaUpdate

    /** Add a new bullet item to a playbook section. */
    @Serializable
    public data class Add(
        override val type: String = "ADD",
        @property:LLMDescription("The content of the bullet item to be added")
        val content: String,
        @property:LLMDescription("The name of the section to which the bullet item should be added")
        val sectionName: String,
        @property:LLMDescription("The short name of the section")
        val sectionShortName: String,
    ) : RawDeltaUpdate() {
        override fun toDeltaUpdate(): DeltaUpdate = DeltaUpdate.Add(content, sectionName, sectionShortName)
    }

    /** Update the content of an existing bullet item. */
    @Serializable
    public data class Update(
        override val type: String = "UPDATE",
        @property:LLMDescription("The ID of the bullet item to be updated")
        val id: String,
        @property:LLMDescription("The new content of the bullet item")
        val newContent: String,
    ) : RawDeltaUpdate() {
        override fun toDeltaUpdate(): DeltaUpdate = DeltaUpdate.Update(id, newContent)
    }

    /** Delete a bullet item from the playbook. */
    @Serializable
    public data class Delete(
        override val type: String = "DELETE",
        @property:LLMDescription("The ID of the bullet item to be deleted")
        val id: String,
    ) : RawDeltaUpdate() {
        override fun toDeltaUpdate(): DeltaUpdate = DeltaUpdate.Delete(id)
    }
}
