package ai.koog.agents.core.optimization.core

/**
 * A training or validation example for optimization.
 *
 * @param TInput The type of the input data provided to the agent.
 * @param TOutput The type of the expected output (label). May be null for unlabeled examples.
 * @property input The input value to be passed to the agent.
 * @property label The expected output for evaluation, or null if unlabeled.
 */
public data class Example<TInput, TOutput>(
    val input: TInput,
    val label: TOutput? = null,
)

/**
 * A collection of training or validation examples.
 */
public typealias Dataset<TInput, TOutput> = List<Example<TInput, TOutput>>
