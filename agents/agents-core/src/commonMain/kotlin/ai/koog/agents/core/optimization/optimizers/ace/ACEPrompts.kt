package ai.koog.agents.core.optimization.optimizers.ace

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import kotlinx.serialization.json.Json

/** Prompt builders for the ACE reflector and curator stages. */
public object ACEPrompts {
    /** Builds the reflector prompt that diagnoses a trajectory and tags playbook bullets. */
    public fun reflectorPrompt(
        groundTruthLabel: String,
        usedPlaybook: ACEPlaybook,
        isSuccess: Boolean,
        trajectoryText: String,
    ): Prompt {
        val systemText = """
Your job is to diagnose the current trajectory: identify what went wrong (or could be better), grounded in execution feedback, and ground truth when applicable.

# Instructions:
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Identify root causes: wrong source of truth, formatting issues, etc. and how to correct them.
- Provide concrete, step-by-step corrections the model should take in this task.
- Be specific about what the model should have done differently
- You will receive bulletpoints that are part of playbook that's used by the model to answer the question.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the original trajectory to generate the correct answer)

# ACE playbook (playbook that's used by model for code generation):
${usedPlaybook.toPromptRepresentation()}

# Outputs:
Your output should be a list of one or many insights. Each insight is a json object, which contains the following fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- errorIdentification: what specifically went wrong in the reasoning?
- rootCauseAnalysis: why did this error occur? What concept was misunderstood?
- correctApproach: what should the model have done instead?
- keyInsight: what strategy, formula, or principle should be remembered to avoid this error?
- bulletTags: a list of json objects with bullet ID and tag for each bulletpoint used by the trajectory

Answer in this exact JSON format:
{
    "insights": [
        {
            "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
            "errorIdentification": "[What specifically went wrong in the reasoning?]",
            "rootCauseAnalysis": "[Why did this error occur? What concept was misunderstood?]",
            "correctApproach": "[What should the model have done instead?]",
            "keyInsight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
            "bulletTags": [
                { "id": "[ID of playbook bullet item]", "tag": "[one of: helpful | harmful | neutral]" }
            ]
        }
    ]
}
        """.trimIndent()
        val userText = """
            AGENT_TRAJECTORY_BEGIN
            $trajectoryText
            AGENT_TRAJECTORY_END

            GROUND_TRUTH_BEGIN
            $groundTruthLabel
            GROUND_TRUTH_END

            IS_TRAJECTORY_SUCCESSFUL: $isSuccess
        """.trimIndent()
        return prompt("ace-reflector") {
            system(systemText)
            user(userText)
        }
    }

    /** Builds the curator prompt that proposes playbook delta updates from reflector insights. */
    public fun curatorPrompt(
        insights: List<TrajectoryInsight>,
        playbook: ACEPlaybook
    ): Prompt {
        val systemText = """
You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

# Context:
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.

# Instructions:
- Review the existing playbook and the reflection from the previous attempt
- Identify the NEW insights, strategies, or mistakes that are MISSING from the current playbook as ADD operations
- Identify INCREMENTAL new additions to existing bullets that generalize well as UPDATE operations
- Identify bullets that consistently HARM trajectories more than help as DELETE operations
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the operations needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable

# Your Task:
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook

Each operation can be one of ONLY the following JSON objects:
{
    "type": "ADD",
    "content": "[The content of the bullet item to be added]",
    "sectionName": "[The name of the section to which the bullet item should be added]",
    "sectionShortName": "[The short name of the section]"
}
OR
{
    "type": "UPDATE",
    "id": "[The id of the bullet item to be updated]",
    "newContent": "[The new content of the bullet item]"
}
OR
{
    "type": "DELETE",
    "id": "[The id of the bullet item to be deleted]"
}

# Sample Output:
{
    "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
    "operations": [
        {
            "type": "ADD",
            "sectionName": "Strategies and Hard Rules",
            "sectionShortName": "strategies",
            "content": "[New reasoning idea...]"
        },
        {
            "type": "DELETE",
            "id": "strategies-5"
        }
    ]
}
        """.trimIndent()
        val userText = """
RECENT_REFLECTION_BEGIN:
${formatInsights(insights)}
RECENT_REFLECTION_END

CURRENT_PLAYBOOK_BEGIN:
${formatPlaybook(playbook)}
CURRENT_PLAYBOOK_END
        """.trimIndent()
        return prompt("ace-curator") {
            system(systemText)
            user(userText)
        }
    }

    private val json = Json { prettyPrint = true }

    private fun formatInsights(insights: List<TrajectoryInsight>): String {
        return json.encodeToString(insights)
    }

    private fun formatPlaybook(playbook: ACEPlaybook): String {
        return playbook.toJson(json)
    }

    /** Renders a playbook as context text for injection into an agent's system prompt. */
    public fun constructPlaybookContext(playbook: ACEPlaybook): String {
        val playbookContent = playbook.toPromptRepresentation()
        if (playbookContent.isBlank()) return ""

        return """
            |ACE Playbook:
            |Read the Playbook first, then execute the task by explicitly leveraging each relevant section:
            |
            |$playbookContent
        """.trimMargin()
    }
}
