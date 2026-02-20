# Input/Output Formatting in OptimizableNode Prompts

This document explains how input and output data is formatted in prompts for
optimizable nodes, how field descriptions flow through the system, and which
flags control the behavior.

## Context

An `OptimizableNode<TInput, TOutput>` builds a prompt from three pieces:
an **instruction** (system message), **demonstrations** (few-shot user/assistant
pairs), and the current **input** (user message). For typed nodes, inputs and
outputs are serialized to JSON. The `@LLMDescription` annotation on serializable
fields provides human-readable descriptions of what each field means.

There are two separate consumers of field descriptions:

1. **The task LLM** — the model that actually executes the node at runtime.
2. **The meta-prompting LLM** — the model used by `InstructionProposer` to
   generate instruction candidates (MIPRO Step 2).

---

## 1. Input Field Descriptions in the Task Prompt (System Message)

**File:** `OptimizablePromptDefaults.kt` — `defaultPromptFn()`

When `includeFieldDescriptions = true`, the default prompt function extracts
`@LLMDescription` annotations from the input serializer's descriptor and
appends them to the system message:

```
<instruction text>

Input fields:
- age: Patient age in years
- cholesterol: Serum cholesterol in mg/dL
```

The user messages contain only the raw JSON value (no descriptions).

**Flag:** `includeFieldDescriptions` parameter on `defaultPromptFn()`.
Default: `false`. Pass `true` to include them:

```kotlin
val myNode by optimizableNode<MyInput, MyOutput>(
    instruction = "...",
    promptFn = defaultPromptFn(
        serializer<MyInput>(),
        serializer<MyOutput>(),
        includeFieldDescriptions = true,
    ),
)
```

**Design choice:** Descriptions are placed only in the system message, not
repeated in each user message. This follows DSPy's approach — field descriptions
appear once in the system prompt, while user/assistant messages contain only
values. This keeps per-example token cost low. The flag defaults to `false`
because for many use cases the field names alone are self-explanatory.

---

## 2. Output Field Descriptions (Structured Output)

**Module:** `prompt-structure`

For typed output nodes, `defaultStructuredExecutePrompt()` calls
`executeStructured()`, which generates a full JSON Schema from the output
serializer. `@LLMDescription` annotations on output fields are embedded as
`"description"` entries in the schema:

```json
{
  "type": "object",
  "properties": {
    "prediction": {
      "type": "string",
      "description": "Heart disease prediction: yes or no"
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score between 0 and 1"
    }
  }
}
```

This schema reaches the LLM via one of two paths:

- **Native structured output** (models that support it): the schema is passed as
  a parameter in the API request (e.g., OpenAI's `response_format`).
- **Manual structured output** (fallback): the schema is rendered as a markdown
  block and appended as a user message, including instructions like
  "Provide ONLY the resulting JSON".

In both cases, `@LLMDescription` annotations on output fields are automatically
included. **There is no flag to disable this** — the descriptions are part of
the JSON Schema specification and are always generated when present.

---

## 3. Field Descriptions in the Instruction Proposer

**Files:** `InstructionProposer.kt`, `agentStrategyExtensions.kt`

The `InstructionProposer` (MIPRO Step 2) uses a meta-LLM to generate
instruction candidates. It provides context about each node via two channels:

### a) `program_code` — Strategy Description

`describeForOptimization()` generates a structural overview of the strategy.
Field descriptions are **always included** here (when present), giving the
meta-LLM context about what data flows through the graph:

```
Strategy: HeartDiseaseStrategy

Optimizable Nodes (3):
  - analyzeRiskFactors
    Current Instruction: "Analyze the patient's risk factors..."
    Input fields:
      - age: Patient age in years
      - cholesterol: Serum cholesterol in mg/dL
    Output fields:
      - riskLevel: Assessed risk level (low/medium/high)
```

### b) `module_code` — Per-Node Description

`buildModuleCodeString()` generates a compact description of a specific node,
including its type signature and field descriptions.

**Flag:** `InstructionProposerConfig.includeFieldDescriptions`.
Default: `true`. Controls whether field descriptions appear in the per-module
code string shown to the meta-LLM.

```kotlin
val config = MIPROv2Config(
    promptModel = myModel,
    proposerConfig = InstructionProposerConfig(
        includeFieldDescriptions = false, // disable for proposer
    ),
)
```

### c) Task Demos in the Proposer

Demonstration examples shown to the meta-LLM use `serializeOrToString()` which
produces raw JSON without field descriptions (matching DSPy's behavior).

---

## 4. JSON Pretty Printing

Different contexts use different pretty-print settings:

| Context | Pretty Print | Rationale |
|---------|:---:|-----------|
| `defaultPromptFn` — task prompt messages | No | Compact JSON minimizes tokens in few-shot user/assistant messages |
| `serializeOrToString` — proposer demos & dataset summary | No* | Used for rendering examples in meta-prompting context |
| Structured output schema | Yes | JSON Schema is rendered as readable markdown |

*Note: `serializeOrToString()` in `optimizersUtils.kt` currently uses
`prettyPrint = false` despite the variable name `prettyJson` and KDoc saying
"pretty-printed". This is a known inconsistency — the proposer context would
benefit from pretty printing for readability, but it's not currently enabled.

**There is no flag for pretty printing.** The JSON format is hardcoded per
context. To change it, supply a custom `promptFn` to `optimizableNode()`.

---

## 5. Summary of Flags

| Flag | Location | Default | Controls |
|------|----------|:-------:|----------|
| `includeFieldDescriptions` | `defaultPromptFn()` | `false` | Input field descriptions in the task prompt system message |
| `includeFieldDescriptions` | `InstructionProposerConfig` | `true` | Field descriptions in per-module code string for the proposer |

Output field descriptions in structured output schemas are always included
(standard JSON Schema behavior, no flag).

Field descriptions in `describeForOptimization()` (strategy-level program code)
are always included (no flag).
