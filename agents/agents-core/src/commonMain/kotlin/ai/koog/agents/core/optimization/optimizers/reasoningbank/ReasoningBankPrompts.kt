package ai.koog.agents.core.optimization.optimizers.reasoningbank

/** Prompt templates for ReasoningBank memory extraction and injection. */
public object ReasoningBankPrompts {
    /** Formats retrieved memory items as context text for injection into an agent's prompt. */
    public fun constructMemoryContext(memories: List<MemoryItem>): String {
        if (memories.isEmpty()) return ""

        val intro = """
            |Below are some memory items that I accumulated from past interaction from the environment
            |that may be helpful to solve the task. You can use it when you feel it's relevant. In each step,
            |please first explicitly discuss if you want to use each memory item or not, and then take action.
        """.trimMargin()

        val items = memories.joinToString("\n\n") { memory ->
            """
            |# Memory item "${memory.title}"
            |## Description: ${memory.description}
            |## Content: ${memory.content}
            """.trimMargin()
        }

        return "$intro\n\n$items"
    }

    /** System prompt for extracting memory items from a successful agent trajectory. */
    public const val EXTRACT_MEMORY_ITEMS_SUCCESS_SYSTEM: String = """
You are an expert in AI agents. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## String Format
Your output must strictly follow the Markdown format shown below:
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task>
"""

    /** System prompt for extracting memory items from a failed agent trajectory. */
    public const val EXTRACT_MEMORY_ITEMS_FAILURE_SYSTEM: String = """
You are an expert in AI agents. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first reflect and think why the trajectory failed, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## String Format
Your output must strictly follow the Markdown format shown below:
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task>
"""
}
