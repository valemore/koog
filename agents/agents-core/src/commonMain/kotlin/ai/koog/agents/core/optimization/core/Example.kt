package ai.koog.agents.core.optimization.core

/**
 * Training or validation data for prompt optimization.
 *
 * An example consists of a map of mapping data field keys (node names) to data field values (node outputs).
 * When generating few-shot examples, the example for the node specified by the field key is converted to a few-shot example.
 * The field values can be any type, strings, data classes, enums, etc., matching whatever the corresponding [OptimizableNode][OptimizableNode]
 *
 * @param TInput The type of the input (matches the strategy's input type).
 * @param TOutput The type of the expected output/label (matches the strategy's output type).
 * @property input The input value for this example.
 * @property label The expected output, or null for unlabeled examples.
 */
public data class Example<TInput, TOutput>(
    val input: TInput,
    val label: TOutput? = null,
) {
    /**
     * Gets the label (expected output) value.
     *
     * @return The label value.
     * @throws IllegalStateException if [labelKey] is null or not present in [data].
     */
    public val label: Any
        get() = labelKey?.let { data[it] }
            ?: error("Example has no labelKey set. Check hasLabel before accessing label.")

    /**
     * Gets the label (expected output) value, or null if unavailable.
     *
     * @return The label value, or null if [labelKey] is null or not present in [data].
     */
    public val labelOrNull: Any?
        get() = labelKey?.let { data[it] }

    /**
     * Checks if this example has a label.
     */
    public val hasLabel: Boolean
        get() = label != null
}

/**
 * Type alias for a dataset (list of examples).
 */
public typealias Dataset<TInput, TOutput> = List<Example<TInput, TOutput>>
