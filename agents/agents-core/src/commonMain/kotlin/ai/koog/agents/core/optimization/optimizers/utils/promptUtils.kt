package ai.koog.agents.core.optimization.optimizers.utils

import ai.koog.prompt.dsl.Prompt

/**
 * Pretty-prints the prompt messages for debugging and trajectory analysis.
 */
public fun Prompt.prettyPrint(): String =
    this.messages
        .withIndex()
        .joinToString("\n") { (i, msg) ->
            "[$i] ${msg.role}:\n${msg.content}\n"
        }
