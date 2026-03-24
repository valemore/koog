package ai.koog.agents.core.optimization.core

/**
 * An evaluation function that scores how well an actual output matches an expected output.
 *
 * Returns a score in the range [0.0, 1.0] where 1.0 indicates a perfect match.
 *
 * @param T The type of the values being compared.
 */
public typealias Metric<T> = (expected: T, actual: T) -> Double
