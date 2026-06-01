# SciClaimEval-Agent

This repository contains an Agentic AI project for scientific claim evaluation. The goal is to build and fine-tune an intelligent agent that can execute tasks described by scientific claim evaluation prompts using the dataset and task definitions provided by the SciClaimEval shared task.

## Project Overview

The project is structured around an agentic pipeline that uses:
- the SciClaimEval dataset and development resources from https://sciclaimeval.github.io/
- candidate models drawn from the available AI services at https://docs.hpc.gwdg.de/services/ai-services/chat-ai/models/index.htm
- task-specific evaluation metrics from the SciClaimEval shared task repository: https://github.com/SciClaimEval/sciclaimeval-shared-task

The final objective is to produce an accurate, robust agent that can understand claim evaluation tasks and generate high-quality outputs for them.

## Phases

### Phase 0: Model Scoring and Baseline Evaluation

- Score available models using the development data from SciClaimEval.
- Compare model performance against baseline metrics.
- Log evaluation results for later candidate model selection.
- This phase is implemented in `phase-0/evaluate-models.py`.

### Phase 1: Candidate Model Selection

- Select the best candidate models for this task based on evaluation metrics from the SciClaimEval shared task.
- Focus on models that perform well on scientific claim reasoning, evidence assessment, and answer accuracy.
- Document the selection criteria and shortlisted models.

### Phase 2: Pipeline Design

- Design the architecture for an agentic pipeline.
- Define the role of each component: prompt generation, reasoning, evidence retrieval, answer validation, and output formatting.
- Ensure the pipeline can be extended for fine-tuning and iterative optimization.

### Phase 3: Optimization and Fine-Tuning

- Optimize each pipeline component for the task.
- Fine-tune the selected models using task-specific training data.
- Evaluate improvements with the SciClaimEval development set and refine until performance goals are met.

## How to Use

1. Review the dataset and task definitions at https://sciclaimeval.github.io/
2. Use the model scoring script in `phase-0/evaluate-models.py` to rank candidate models.
3. Select candidate models and design the agent pipeline.
4. Fine-tune and optimize using the evaluation metrics from https://github.com/SciClaimEval/sciclaimeval-shared-task.

## Expected Outcomes

- A ranked list of candidate models for claim evaluation tasks.
- A designed and documented agent pipeline for the SciClaimEval task.
- Fine-tuned model components that improve accuracy and task execution performance.
- A reproducible evaluation workflow using SciClaimEval development data.

## Notes

- This repository is intended for experimentation with agentic model-based pipelines for scientific claim verification and evaluation.
- The project is work-in-progress and may expand with additional scripts, evaluation tools, and fine-tuning utilities.

## References

- SciClaimEval website: https://sciclaimeval.github.io/
- HPC AI models list: https://docs.hpc.gwdg.de/services/ai-services/chat-ai/models/index.htm
- SciClaimEval shared task: https://github.com/SciClaimEval/sciclaimeval-shared-task
