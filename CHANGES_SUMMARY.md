# Data Flywheel FSI - Changes Summary

## 1. NeMo Microservices Platform Deployment (`scripts/deploy-nmp.sh`)

### Increased Minikube Disk Space
- **Line 477-484**: Added `--disk-size=500g` to minikube start command
- **Line 500**: Updated systemd service to persist disk size setting across restarts
- **Reason**: Prevent "no space left on device" errors during container image pulls

### Updated Helm Chart Management
- **Lines 18-19**: Changed from hardcoded URL to repository-based chart fetching
  - `HELM_CHART_REPO="nvidia/nemo-microservices-helm-chart"`
  - `HELM_CHART_VERSION=""` (empty = latest)
- **Lines 623-625, 641-643**: Modified helm fetch/install to use repo reference with optional version
- **Reason**: Automatically use latest Helm chart instead of fixed version

---

## 2. Generic Workload Support (Chat Completion / Classification)

### Added Evaluation Configuration (`src/config.py`, `config/config.yaml`)

**src/config.py (Lines 109-119)**:
```python
class EvaluationConfig(BaseModel):
    workload_type: Literal["auto", "generic", "tool_calling"] = "auto"
    tool_eval_type: Literal["tool-calling-metric", "tool-calling-judge"] = "tool-calling-metric"
```

**config/config.yaml (Lines 78-89)**:
```yaml
evaluation_config:
  workload_type: "auto"  # auto-detect, generic, or tool_calling
  tool_eval_type: "tool-calling-metric"
```
- **Reason**: Allow explicit control over evaluation strategy instead of implicit behavior

### Workload Type Detection (`src/lib/flywheel/util.py`)

**Lines 81-114**: Added `config_override` parameter to `identify_workload_type()`
- Respects explicit config setting when provided
- Falls back to auto-detection based on `tool_calls` presence
- **Reason**: Support both auto-detection and manual override for evaluation type

### Score Extraction Fixes

**src/tasks/tasks.py (Lines 615-620)**:
```python
elif previous_result.workload_type == WorkloadClassification.GENERIC:
    if results["tasks"]["chat-completion"]:
        scores["f1_score"] = results["tasks"]["chat-completion"]["metrics"]["f1"]["scores"]["f1_score"]["value"]
else:
    raise ValueError(f"Unsupported workload type: {previous_result.workload_type}")
```
- **Changed from**: Trying to access `llm-as-judge` task (which doesn't exist for generic workloads)
- **Changed to**: Extract F1 score from `chat-completion` task
- **Reason**: Generic workloads use F1 score for classification, not LLM-as-judge

**src/lib/integration/mlflow_client.py**:
- **Lines 220-224**: Changed task_key from `"llm-as-judge"` to `"chat-completion"` for generic workloads
- **Lines 144-156**: Extract `f1_score` from `f1` metrics instead of `similarity` from `llm-judge`
- **Lines 283-291**: Updated metric columns to include `"f1_score"` instead of `"similarity"`
- **Reason**: Align MLflow logging with correct evaluation metrics for classification tasks

---

## 3. Monitoring & Visualization (`notebooks/utils/job_monitor_helper.py`)

### Enhanced Score Display
**Lines 61-63**: Added explicit F1 score extraction
```python
if "f1_score" in all_scores:
    row["F1 Score"] = all_scores["f1_score"]
```

### Adaptive Plotting Logic
**Lines 138-198**: Updated plot generation to support both workload types
- **Tool Calling**: 3 metrics (function name accuracy, exact match, LLM-judge)
- **Generic**: 1 metric (F1 Score)
- **Grouped bar chart**: X-axis = Models, Grouped bars = BASE-EVAL vs CUSTOMIZED-EVAL
- Auto-detects available metrics and adapts visualization accordingly

**Key Changes**:
- Detects metric type from available columns (lines 140-157)
- Creates grouped bar chart with models on x-axis (lines 159-198)
- Maintains consistent visual format across both workload types
- **Reason**: Single codebase supporting both tool calling and classification workloads

---

## Summary of Key Improvements

1. **Infrastructure**: Increased disk space allocation for Minikube to handle large container images
2. **Flexibility**: Added configuration-driven evaluation strategy selection
3. **Correctness**: Fixed score extraction logic to use appropriate metrics per workload type
4. **Visualization**: Unified plotting code that adapts to tool calling or classification metrics
5. **Maintainability**: Explicit error handling for unsupported workload types

All changes maintain backward compatibility with existing tool calling workloads while enabling support for generic classification tasks.

