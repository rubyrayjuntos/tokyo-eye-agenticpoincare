# Requirements Document

## Introduction

The Hypothesis Engine gives the Tokyo Eye agent a structured framework for generating, tracking, testing, and validating scientific hypotheses about protein structures. Rather than simply reporting numerical findings, the agent proposes testable hypotheses, gathers evidence using existing pipeline tools, and tracks hypothesis status through a formal lifecycle. Guardrails enforce scientific rigor: falsifiability, evidence balance, confidence decay, and contradiction detection.

## Glossary

- **Hypothesis_Engine**: The subsystem responsible for creating, storing, testing, and evaluating scientific hypotheses about protein structures.
- **Hypothesis**: A structured scientific claim about a protein structure, containing a statement, optional mechanism, testable predictions, gathered evidence, and a lifecycle status.
- **Prediction**: A testable claim derived from a hypothesis that can be evaluated by calling an existing agent tool with specific parameters and comparing the result against a threshold.
- **Evidence**: A record of data gathered (from tool calls or user input) that either supports or contradicts a hypothesis, with an associated strength weight.
- **Confidence_Score**: A floating-point value between 0.0 and 1.0 representing the degree to which gathered evidence supports a hypothesis (0.5 = no evidence / maximum uncertainty).
- **Hypothesis_Status**: One of: `proposed`, `gathering`, `supported`, `contradicted`, `inconclusive`.
- **Guardrail**: A validation rule enforcing scientific rigor on hypothesis creation and evaluation.
- **Normalizer**: The single governed write path (`data/normalizer/core.py`) through which all data writes must pass.
- **Agent**: The LLM-driven coordinator that orchestrates tool calls and scientific reasoning.

## Requirements

### Requirement 1: Hypothesis Creation

**User Story:** As a researcher, I want the agent to propose structured hypotheses with testable predictions, so that scientific reasoning is explicit and traceable.

#### Acceptance Criteria

1. WHEN the Agent proposes a hypothesis, THE Hypothesis_Engine SHALL create a Hypothesis record with a unique ID, structure_id, statement, status of `proposed`, and confidence of 0.5
2. WHEN a Hypothesis is created without at least one Prediction that could contradict it, THE Hypothesis_Engine SHALL reject the creation and return an error explaining the falsifiability requirement
3. WHEN a Hypothesis is created, THE Hypothesis_Engine SHALL persist it through the Normalizer write path
4. WHEN a Hypothesis is created with Predictions, THE Hypothesis_Engine SHALL store each Prediction with a reference to the tool and parameters needed to test it

### Requirement 2: Prediction Testing

**User Story:** As a researcher, I want the agent to automatically test hypothesis predictions using existing pipeline tools, so that hypotheses are evaluated with real data.

#### Acceptance Criteria

1. WHEN the Agent tests a Hypothesis, THE Hypothesis_Engine SHALL execute each Prediction by calling the specified tool with the stored parameters
2. WHEN a Prediction tool call returns a result, THE Hypothesis_Engine SHALL compare the result against the Prediction threshold and mark the Prediction as passed or failed
3. WHEN all Predictions for a Hypothesis have been tested, THE Hypothesis_Engine SHALL create Evidence records from the results and update the Hypothesis confidence score
4. WHEN a Prediction references a tool that does not exist or returns an error, THE Hypothesis_Engine SHALL mark that Prediction as untestable and exclude it from confidence calculation

### Requirement 3: Evidence Management

**User Story:** As a researcher, I want to add manual evidence to hypotheses and have confidence recalculated, so that external observations can inform hypothesis status.

#### Acceptance Criteria

1. WHEN evidence is added to a Hypothesis, THE Hypothesis_Engine SHALL store the Evidence record with source, supports/contradicts flag, strength, and description
2. WHEN evidence is added, THE Hypothesis_Engine SHALL recalculate the Hypothesis confidence score using all gathered evidence
3. THE Hypothesis_Engine SHALL calculate confidence as the ratio of supporting evidence strength to total evidence strength, returning 0.5 when no evidence exists

### Requirement 4: Hypothesis Lifecycle

**User Story:** As a researcher, I want hypotheses to transition through a clear lifecycle based on evidence, so that I can track the state of scientific reasoning.

#### Acceptance Criteria

1. WHEN a Hypothesis confidence exceeds 0.7 and at least 2 pieces of evidence exist, THE Hypothesis_Engine SHALL transition the status to `supported`
2. WHEN a Hypothesis confidence drops below 0.3 and at least 2 pieces of evidence exist, THE Hypothesis_Engine SHALL transition the status to `contradicted`
3. WHEN a Hypothesis has been in `gathering` status for more than 7 days without new evidence, THE Hypothesis_Engine SHALL decay the confidence toward 0.5 by 10% per day
4. WHEN the Agent begins testing predictions for a Hypothesis, THE Hypothesis_Engine SHALL transition the status from `proposed` to `gathering`

### Requirement 5: Hypothesis Retrieval

**User Story:** As a researcher, I want to list and filter hypotheses by structure or status, so that I can review the current state of scientific reasoning.

#### Acceptance Criteria

1. WHEN a user requests hypotheses for a structure, THE Hypothesis_Engine SHALL return all Hypothesis records for that structure_id with their current status, confidence, and evidence summary
2. WHEN a user filters hypotheses by status, THE Hypothesis_Engine SHALL return only hypotheses matching the requested status
3. WHEN returning hypothesis details, THE Hypothesis_Engine SHALL include the count of supporting and contradicting evidence and the list of Predictions with their pass/fail status

### Requirement 6: Contradiction Detection

**User Story:** As a researcher, I want to be alerted when new pipeline results contradict an active hypothesis, so that I can reassess my scientific reasoning.

#### Acceptance Criteria

1. WHEN new pipeline results are produced for a structure with active hypotheses (status `supported` or `gathering`), THE Hypothesis_Engine SHALL re-evaluate relevant Predictions against the new data
2. WHEN re-evaluation shows a previously-passing Prediction now fails, THE Hypothesis_Engine SHALL add contradicting Evidence and alert the user with a description of the contradiction

### Requirement 7: Confidence Calculation

**User Story:** As a researcher, I want hypothesis confidence to be calculated transparently from evidence, so that I can understand and trust the scoring.

#### Acceptance Criteria

1. THE Hypothesis_Engine SHALL calculate confidence as: supporting_strength / (supporting_strength + contradicting_strength), where strength is the sum of evidence strength values for each category
2. WHEN no evidence exists for a Hypothesis, THE Hypothesis_Engine SHALL return a confidence of 0.5
3. WHEN all evidence has zero total strength, THE Hypothesis_Engine SHALL return a confidence of 0.5

### Requirement 8: Serialization

**User Story:** As a developer, I want hypothesis data to be serializable to and from JSON for API responses and storage, so that the data model integrates with the existing system.

#### Acceptance Criteria

1. THE Hypothesis_Engine SHALL serialize Hypothesis objects (including nested Predictions and Evidence) to JSON
2. THE Hypothesis_Engine SHALL deserialize valid JSON back into equivalent Hypothesis objects (round-trip)
3. THE Hypothesis_Engine SHALL produce a pretty-printed JSON representation of Hypothesis objects for API responses
