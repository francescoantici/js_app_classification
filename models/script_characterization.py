from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union

class ScriptCharacterization(BaseModel):
    """
    Base class to model the output dto of the LLM
    """
    application: str = Field(description = "The name of the application executed by the commands.", max_length=25)


class Orchestration(BaseModel):
    type: Literal["workflow_managed", "job_array", "singular"] = Field(description="How the script orchestrates its work.")
    tool_detected: Optional[str] = Field(description="Orchestration tool detected in the script, or null if none.")


class ExecutionEnvironment(BaseModel):
    type: Literal["containerized", "virtual_environment", "bare_metal"] = Field(description="Execution environment of the script.")
    tool_detected: Optional[str] = Field(description="Environment tooling detected in the script (e.g., Docker, Singularity, conda), or null if none.")


class Workload(BaseModel):
    primary_software: Optional[str] = Field(description="Main software package executed by the script, or null if unknown.")
    target_executable: Optional[str] = Field(description="Primary executable or entrypoint invoked, or null if unknown.")
    workload_category: Optional[Literal["ai_ml", "bioinformatics", "simulation_modeling", "data_processing_etl", "visualization", "software_build", "other_unknown"]] = Field(description="Category of the workload, or null if unknown.")


class StepOrchestration(BaseModel):
    type: Literal["workflow_managed", "job_array", "standalone"] = Field(description="How the step is orchestrated.")
    tool_detected: Optional[str] = Field(description="Orchestration tool detected for the step, or null if none.")


class StepExecutionEnvironment(BaseModel):
    type: Literal["containerized", "native"] = Field(description="Execution environment of the step.")
    tool_detected: Optional[str] = Field(description="Environment tooling detected for the step (e.g., Docker, Apptainer, conda), or null if none.")


class WorkloadClass(BaseModel):
    type: Literal["simulation", "ai_ml", "data_analytics", "utility"] = Field(description="Class of the workload executed by the step.")
    primary_software: Optional[str] = Field(description="Main software package executed by the step (e.g., 'GROMACS', 'python', 'apptainer'), or null if unknown.")
    executed_file: Optional[str] = Field(description="File executed by the step, or null if unknown.")
    parameters: Optional[List[str]] = Field(description="Parameters passed to the executed software, or null if unknown.")
    step_description: str = Field(description="Description of what the step does.")


class Step(BaseModel):
    step_id: int = Field(description="Identifier of the step within the script.")
    orchestration: StepOrchestration
    execution_environment: StepExecutionEnvironment
    workload_class: WorkloadClass


class ScriptStepsCharacterization(BaseModel):
    """
    Models the output dto of the step-based LLM characterization.
    """
    description: str = Field(description="Description of the overall purpose of the script.")
    domain: Union[str, List[str]] = Field(description="Scientific or application domain(s) of the script.")
    steps: List[Step] = Field(description="Ordered list of steps performed by the script.")


class ScriptTaxonomyCharacterization(BaseModel):
    """
    Models the output dto of the taxonomy-based LLM characterization.
    """
    orchestration: Orchestration
    execution_environment: ExecutionEnvironment
    workload: Workload
    compute_profile: List[str] = Field(description="Tags describing the computational profile of the job.")
