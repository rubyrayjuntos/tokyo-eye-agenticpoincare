"""
Benchmark API Endpoints

Provides REST API for benchmark validation pipeline execution and reporting.

Validates: Requirements 14.1-14.8
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List

from gosp.services.benchmark_validation import (
    run_benchmark_validation,
    BenchmarkReport,
    ClassAResult,
    ClassBResult,
    ClassCResult,
    ClassDResult,
    ClassEResult,
    ClassGResult,
)


router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


class BenchmarkExecutionResponse(BaseModel):
    """Response model for benchmark execution."""
    report: BenchmarkReport = Field(..., description="Consolidated benchmark report")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "report": {
                    "execution_timestamp": "2026-02-13T14:23:01Z",
                    "total_structures": 30,
                    "total_passed": 25,
                    "total_failed": 5,
                    "success_rate": 83.33,
                    "class_a_results": [],
                    "class_b_results": [],
                    "class_c_results": [],
                    "class_d_results": [],
                    "class_e_results": [],
                    "class_g_results": [],
                    "execution_time_sec": 7200.0,
                    "errors": []
                }
            }
        }
    )


@router.post("/execute", response_model=BenchmarkExecutionResponse)
async def execute_benchmark() -> BenchmarkExecutionResponse:
    """
    Execute complete benchmark validation pipeline.
    
    This endpoint:
    - Runs validation on 30 structures across 7 classes
    - Validates Class A structures against Fernández literature (±2 wrapping count)
    - Validates Class B structures with HDX-MS correlation (R² ≥ 0.50)
    - Validates Class C antibodies against clinical ADA data (≥60% overlap)
    - Validates Class D enzyme voids against POVME3 (≤20% error)
    - Validates Class E linkers with SAXS/FRET data (within 1 order of magnitude)
    - Flags Class G negative controls as out-of-scope (100% detection)
    - Generates consolidated benchmark report with success metrics
    
    The benchmark pipeline integrates all backend services:
    - Structure ingestion
    - Dehydron detection
    - Void detection
    - HDX-MS validation
    - Immunogenicity screening
    
    Returns:
        BenchmarkExecutionResponse with consolidated report
    
    Raises:
        HTTPException: If benchmark execution fails
    
    Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8
    """
    try:
        # Run benchmark validation
        report = await run_benchmark_validation()
        
        return BenchmarkExecutionResponse(report=report)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "BenchmarkExecutionError",
                "message": f"Benchmark execution failed: {str(e)}",
                "details": {},
                "suggested_action": "Check system logs and try again"
            }
        )


@router.get("/report", response_model=BenchmarkExecutionResponse)
async def get_latest_benchmark_report() -> BenchmarkExecutionResponse:
    """
    Get the latest benchmark validation report.
    
    This endpoint returns the most recently executed benchmark report.
    In a production system, this would retrieve the report from persistent storage.
    
    Returns:
        BenchmarkExecutionResponse with latest report
    
    Raises:
        HTTPException: If no report is available
    """
    # In production, this would retrieve from database
    # For now, return a placeholder
    raise HTTPException(
        status_code=404,
        detail={
            "error": True,
            "error_type": "ReportNotFound",
            "message": "No benchmark report available. Execute benchmark first.",
            "details": {},
            "suggested_action": "Call POST /api/benchmark/execute to generate a report"
        }
    )
