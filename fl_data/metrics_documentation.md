# Database Pair Annotation Metrics Documentation

## Overview

This document describes the manual annotation metrics used to evaluate database pairs for federated learning. **All metrics are manually reviewed and annotated by human evaluators** - they are not automatically calculated from the data.

## Metrics

### 1. `is_valid_pair` (Valid Pair)

**Purpose**: Indicates whether a database pair is valid for federated learning.

**Annotation Guidelines**:
- **true**: The database pair is considered valid and suitable for federated learning. The two databases should be semantically related, have compatible schemas, and be appropriate for joint training.
- **false**: The database pair is not valid for federated learning. This could be due to incompatible schemas, unrelated domains, or other factors that make federated learning inappropriate.

**Considerations**:
- Review the similarity score, common columns, and overall compatibility
- Consider whether the databases represent related domains or concepts
- Evaluate if federated learning would be meaningful for this pair

---

### 2. `column_alignment_good` (Column Alignment)

**Purpose**: Evaluates whether the common columns between two databases are well-aligned.

**Annotation Guidelines**:
- **true**: The common columns are well-aligned. Column names are semantically similar, column types are compatible, and the columns represent the same or closely related concepts across both databases.
- **false**: The common columns are poorly aligned. Column names may be semantically different despite matching, column types may be incompatible, or the columns may represent different concepts.

**Considerations**:
- Review the list of common columns and their types
- Assess semantic similarity of column names
- Evaluate type compatibility (e.g., numeric vs categorical)
- Consider whether columns represent the same concepts in both databases

**Reference Information**:
- Common columns: See "Column Information" section
- Column types: See "Column Types" section

---

### 3. `label_reasonable` (Label Reasonableness)

**Purpose**: Evaluates whether the selected label column is appropriate and reasonable for the learning task.

**Annotation Guidelines**:
- **true**: The label column selection is reasonable. The chosen label is meaningful, relevant to the task, and appropriate for prediction. It represents a valid learning objective that makes sense in the context of the database pair.
- **false**: The label column selection is not reasonable. The chosen label may be inappropriate, irrelevant, or not suitable as a prediction target. It may not represent a meaningful learning objective for this database pair.

**Considerations**:
- Is the selected label column meaningful and relevant to the task?
- Does the label represent a valid prediction objective?
- Is the label appropriate for the domain and use case?
- Would predicting this label be useful or meaningful?
- Consider whether there might be better label choices available

**Reference Information**:
- Label column: See "Basic Information" section
- Task type: See "Basic Information" section (classification or regression)
- Available columns: Review all columns to understand what labels could be selected

---

### 4. `task_reasonable` (Task Reasonableness)

**Purpose**: Evaluates whether the learning task setup and configuration are reasonable and appropriate.

**Annotation Guidelines**:
- **true**: The task setup is reasonable. The task type (classification or regression) is appropriate for the data and use case. The task definition makes sense and aligns with the domain requirements. The overall task configuration is sensible.
- **false**: The task setup is not reasonable. The task type may be inappropriate (e.g., using classification when regression would be better, or vice versa). The task definition may not align well with the data characteristics or domain needs. The overall task configuration may be problematic.

**Considerations**:
- Is the task type (classification vs regression) appropriate for the data and use case?
- Does the task definition make sense in the context of the database pair?
- Is the task configuration aligned with domain requirements?
- Would this task setup lead to meaningful learning outcomes?
- Consider whether the task type matches the nature of the label and features

**Reference Information**:
- Task type: See "Basic Information" section (classification or regression)
- Label column: See "Basic Information" section
- Feature columns: See "Column Information" section

---

### 5. `is_related` (Relatedness)

**Purpose**: Evaluates whether two databases are semantically related.

**Annotation Guidelines**:
- **true**: The databases are highly related. They represent similar domains, concepts, or use cases. The similarity score is high, and there are sufficient common columns indicating semantic overlap.
- **false**: The databases are weakly related or unrelated. Despite having some common columns, they may represent different domains or concepts, making them less suitable for federated learning.

**Considerations**:
- Review the similarity score between databases
- Assess semantic relationship based on database descriptions/abstracts
- Evaluate common columns for semantic overlap
- Consider domain compatibility

**Reference Information**:
- Similarity score: See "Basic Information" section
- Common columns: See "Column Information" section
- Database IDs: See "Basic Information" section (can help identify domains)

---

## Annotation Workflow

1. **Review Basic Information**: Check database IDs, similarity score, task type, label column
2. **Examine Column Information**: Review common columns, feature columns, column types to understand schema alignment
3. **Assess Label Selection**: Evaluate whether the selected label column is appropriate and meaningful for the learning task
4. **Evaluate Task Configuration**: Assess whether the task type (classification/regression) and overall task setup are reasonable
5. **Assess Database Relatedness**: Review similarity score and common columns to evaluate semantic relationship
6. **Make Annotation Decisions**: For each metric, decide true/false based on the guidelines above, focusing on appropriateness and reasonableness rather than data quality metrics
7. **Final Validation**: Review `is_valid_pair` considering all other metrics to make the final decision

## Best Practices

- **Focus on Appropriateness**: Evaluate whether choices and configurations are appropriate, not whether data meets technical thresholds
- **Consistency**: Apply the same criteria across all database pairs
- **Context Awareness**: Consider the specific domain and use case when evaluating reasonableness
- **Holistic Evaluation**: Look at all metrics together when making final decisions
- **Judgment-Based**: Remember these are human judgments about appropriateness, not automated calculations based on data statistics

## Notes

- **All metrics are manual annotations** - they reflect human judgment about appropriateness and reasonableness, not automated calculations based on data statistics
- **Focus on choices and configurations**: These metrics evaluate whether selections (label column, task type) and configurations are appropriate, not whether the data meets technical quality thresholds
- Metrics can be updated or revised as annotation progresses
- Empty values indicate that a metric has not been annotated yet
- The `is_valid_pair` metric is the primary decision metric, while other metrics provide supporting information for that decision
- When in doubt, consider: "Is this choice/configuration reasonable and appropriate?" rather than "Does this data meet technical criteria?"
