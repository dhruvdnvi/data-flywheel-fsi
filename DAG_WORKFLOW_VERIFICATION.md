# Data Flywheel DAG Workflow Verification

## Complete Workflow Structure

### Main DAG Entry Point: `run_nim_workflow_dag()`

```
run_nim_workflow_dag()
  └─ workflow = chain(
       1. initialize_workflow
       2. dataset_workflow
       3. wait_for_llm_as_judge
       4. chain(*nim_chains)
       5. finalize_flywheel_run
     )
```

## Detailed Workflow Breakdown

### 1. Initialize Workflow
**Task:** `initialize_workflow()`
**Purpose:** Set up the flywheel run and create NIM records in database
**Actions:**
- Updates flywheel run status to RUNNING
- Creates LLM judge run record
- Creates NIM run records for each candidate model (pending status)

### 2. Dataset Workflow
**Task:** `create_datasets()`
**Purpose:** Load data, split, validate, and upload to NeMo datastore
**Actions:**
- Export records from Elasticsearch
- Identify workload type (TOOL_CALLING or GENERIC)
- Validate and clean records
- Split into eval/train/val sets
- Format for evaluation and training
- Upload BASE dataset (eval data)
- Upload TRAIN dataset (train + val data)
- **NO LONGER:** ICL dataset creation or embedding setup

**Returns:** 
```python
{
    DatasetType.BASE: "flywheel-eval-...",
    DatasetType.TRAIN: "flywheel-train-..."
}
```

### 3. Wait for LLM as Judge
**Task:** `wait_for_llm_as_judge()`
**Purpose:** Ensure LLM judge is ready for evaluations
**Actions:**
- If remote: skip (already available)
- If local: wait for model deployment and sync

### 4. NIM Chains (Parallel)
For each NIM in config (e.g., llama-3.2-1b, llama-3.2-3b):
```
nim_chain = chain(
    a. spin_up_nim
    b. group(
         i. run_base_eval
         ii. chain(
               - start_customization
               - run_customization_eval
            )
       )
    c. shutdown_deployment
)
```

#### 4a. Spin Up NIM
**Task:** `spin_up_nim(nim_config)`
**Purpose:** Deploy the base NIM model
**Actions:**
- Deploy NIM via DMS client
- Wait for model sync
- Update NIM status to READY

#### 4b.i. Run Base Evaluation
**Task:** `run_base_eval()` → calls `run_generic_eval(EvalType.BASE, DatasetType.BASE)`
**Purpose:** Evaluate base model performance
**Actions:**
1. Get evaluator with judge config
2. Determine eval config based on workload type:
   - **TOOL_CALLING**: `get_tool_calling_config()` → tool calling metrics
   - **GENERIC**: `get_chat_completion_config()` → F1 score metrics ✓
3. Run evaluation job via NeMo Evaluator
4. Wait for completion with progress updates
5. Fetch and store results

**Evaluation Config for GENERIC:**
```python
{
    "type": "custom",
    "tasks": {
        "chat-completion": {
            "type": "chat-completion",
            "dataset": "hf://datasets/dfwbp/flywheel-eval-.../eval_data.jsonl",
            "params": {"template": {"messages": "{{ item.request.messages | tojson }}"}},
            "metrics": {
                "f1": {
                    "type": "f1",
                    "params": {
                        "ground_truth": "{{ item.response.choices[0].message.content | trim }}"
                    }
                }
            }
        }
    }
}
```

#### 4b.ii. Customization Chain

**Task 1:** `start_customization()`
**Purpose:** Fine-tune model with LoRA
**Actions:**
- Create customization job via NeMo Customizer
- Train on TRAIN dataset
- Wait for completion
- Get customized model name (e.g., "meta/llama-3.2-1b-instruct@job-abc123")

**Task 2:** `run_customization_eval()` → calls `run_generic_eval(EvalType.CUSTOMIZED, DatasetType.BASE)`
**Purpose:** Evaluate customized model performance
**Actions:**
- Same as base eval but uses customized model
- Compares F1 scores: customized vs base

#### 4c. Shutdown Deployment
**Task:** `shutdown_deployment()`
**Purpose:** Clean up NIM resources
**Actions:**
- Shutdown NIM deployment
- Update NIM run status to COMPLETED

### 5. Finalize Flywheel Run
**Task:** `finalize_flywheel_run()`
**Purpose:** Update flywheel run status and metrics
**Actions:**
- Calculate aggregate metrics
- Update flywheel run status to COMPLETED
- Log to MLflow if enabled

## Removed Components (ICL Elimination)

### ❌ Removed from Workflow:
1. **Embedding NIM spin-up** in dataset workflow
2. **ICL dataset creation** with example injection
3. **ICL evaluation task** (`run_icl_eval`)
4. **Embedding index cleanup** in finally blocks

### ✓ Kept and Modified:
1. **Base evaluation** - now uses F1 scores for GENERIC workloads
2. **Customization evaluation** - now uses F1 scores for GENERIC workloads
3. **Dataset creation** - simplified to BASE + TRAIN only

## Workload Type Routing in Evaluator

```python
def run_evaluation(workload_type, tool_eval_type, ...):
    if workload_type == WorkloadClassification.TOOL_CALLING:
        if tool_eval_type == ToolEvalType.TOOL_CALLING_METRIC:
            config = self.get_tool_calling_config()
        elif tool_eval_type == ToolEvalType.TOOL_CALLING_JUDGE:
            config = self.get_tool_llm_as_judge_config()
        else:
            raise ValueError(f"Unsupported tool eval type: {tool_eval_type}")
    elif workload_type == WorkloadClassification.GENERIC:
        config = self.get_chat_completion_config()  # F1 scores
    else:
        raise ValueError(f"Unsupported workload type: {workload_type}")
```

## Key Verification Points

✅ **DAG Structure:** Correct chain and group usage
✅ **Embedding Removal:** No longer spins up embedding NIM
✅ **ICL Task Removal:** `run_icl_eval` not in workflow
✅ **Dataset Simplification:** Only BASE and TRAIN datasets
✅ **Evaluator Routing:** Explicit GENERIC handling with F1 scores
✅ **Error Handling:** Fail-fast for unsupported workload types
✅ **Pattern Consistency:** Chat completion follows tool calling pattern

## Summary

The workflow is now streamlined:
- **2 evaluation types** per NIM (BASE + CUSTOMIZED) instead of 3
- **2 datasets** created (BASE + TRAIN) instead of 3
- **No embeddings** or ICL logic
- **F1 scores** for generic/classification workloads
- **Explicit routing** with proper error handling
