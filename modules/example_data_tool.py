"""
Example Data Processing Tool Module
Demonstrates how to create a modular tool using the new module system.

@version: 1.2.0
@author: AgentKen Team
@id: data_processor_tool
"""

import json
import csv
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from module_system import ToolModule, ModuleCapability, ModuleMetadata, ModuleType, ModuleDependency, DependencyType


class DataProcessorToolModule(ToolModule):
    """Modular data processing tool with various data manipulation capabilities"""
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        super().__init__(module_id, config)
        self.max_file_size = config.get("max_file_size", 100 * 1024 * 1024) if config else 100 * 1024 * 1024  # 100MB
        self.supported_formats = config.get("supported_formats", ["csv", "json", "xlsx", "parquet"]) if config else ["csv", "json", "xlsx", "parquet"]
        self.temp_dir = Path(config.get("temp_dir", "temp")) if config else Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize metadata
        self.metadata = ModuleMetadata(
            id="data_processor_tool",
            name="Data Processor Tool",
            version="1.2.0",
            module_type=ModuleType.TOOL,
            description="Comprehensive data processing tool with cleaning, transformation, and analysis capabilities",
            author="AgentKen Team",
            license="MIT",
            homepage="https://agentken.io/modules/data-processor",
            documentation="https://docs.agentken.io/modules/data-processor",
            
            dependencies=[
                ModuleDependency(
                    name="file_manager_tool",
                    version="1.0.0",
                    dependency_type=DependencyType.OPTIONAL,
                    description="Enhanced file management capabilities"
                )
            ],
            
            python_requirements=[
                "pandas>=1.3.0",
                "numpy>=1.21.0",
                "openpyxl>=3.0.0",
                "pyarrow>=5.0.0"
            ],
            
            capabilities=[
                ModuleCapability(
                    name="load_data",
                    description="Load data from various file formats",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to data file"},
                            "format": {"type": "string", "enum": ["csv", "json", "xlsx", "parquet"]},
                            "options": {"type": "object", "description": "Format-specific options"}
                        },
                        "required": ["file_path"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "description": "Loaded data"},
                            "shape": {"type": "array", "items": {"type": "integer"}},
                            "columns": {"type": "array", "items": {"type": "string"}},
                            "dtypes": {"type": "object"},
                            "load_time": {"type": "number"}
                        }
                    },
                    tags=["data", "load", "file"]
                ),
                
                ModuleCapability(
                    name="clean_data",
                    description="Clean and preprocess data",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "description": "Data to clean"},
                            "operations": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["remove_duplicates", "handle_missing", "normalize", "outliers"]}
                            },
                            "options": {"type": "object"}
                        },
                        "required": ["data"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "cleaned_data": {"type": "array"},
                            "cleaning_report": {"type": "object"},
                            "rows_removed": {"type": "integer"},
                            "processing_time": {"type": "number"}
                        }
                    },
                    tags=["data", "cleaning", "preprocessing"]
                ),
                
                ModuleCapability(
                    name="transform_data",
                    description="Transform and manipulate data",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "description": "Data to transform"},
                            "transformations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["filter", "sort", "group", "aggregate", "pivot"]},
                                        "parameters": {"type": "object"}
                                    }
                                }
                            }
                        },
                        "required": ["data", "transformations"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "transformed_data": {"type": "array"},
                            "transformation_log": {"type": "array"},
                            "processing_time": {"type": "number"}
                        }
                    },
                    tags=["data", "transformation", "manipulation"]
                ),
                
                ModuleCapability(
                    name="analyze_data",
                    description="Perform statistical analysis on data",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "description": "Data to analyze"},
                            "analysis_type": {"type": "string", "enum": ["descriptive", "correlation", "distribution", "outliers"]},
                            "columns": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["data", "analysis_type"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "analysis_results": {"type": "object"},
                            "statistics": {"type": "object"},
                            "visualizations": {"type": "array"},
                            "insights": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    tags=["data", "analysis", "statistics"]
                ),
                
                ModuleCapability(
                    name="export_data",
                    description="Export data to various formats",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "description": "Data to export"},
                            "file_path": {"type": "string", "description": "Output file path"},
                            "format": {"type": "string", "enum": ["csv", "json", "xlsx", "parquet"]},
                            "options": {"type": "object"}
                        },
                        "required": ["data", "file_path", "format"]
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "file_size": {"type": "integer"},
                            "rows_exported": {"type": "integer"},
                            "export_time": {"type": "number"}
                        }
                    },
                    tags=["data", "export", "file"]
                )
            ],
            
            configuration_schema={
                "type": "object",
                "properties": {
                    "max_file_size": {"type": "integer", "minimum": 1024, "maximum": 1073741824},
                    "supported_formats": {"type": "array", "items": {"type": "string"}},
                    "temp_dir": {"type": "string"},
                    "memory_limit": {"type": "integer"},
                    "parallel_processing": {"type": "boolean"}
                }
            },
            
            tags=["data", "processing", "analysis", "tool"],
            category="data",
            priority=2
        )
    
    def initialize(self) -> bool:
        """Initialize the data processor tool module"""
        try:
            # Validate configuration
            if not self.validate_config(self.config):
                self.logger.error("Invalid configuration")
                return False
            
            # Check required dependencies
            try:
                import pandas
                import numpy
                self.logger.info("Required dependencies available")
            except ImportError as e:
                self.logger.error(f"Missing required dependency: {e}")
                return False
            
            # Create temp directory
            self.temp_dir.mkdir(exist_ok=True)
            
            self.logger.info("Data processor tool module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data processor tool: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the data processor tool module"""
        try:
            # Clean up temporary files
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            self.logger.info("Data processor tool module shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown data processor tool: {e}")
            return False
    
    def get_capabilities(self) -> List[ModuleCapability]:
        """Return list of capabilities"""
        return self.metadata.capabilities
    
    def execute(self, capability: str, **kwargs) -> Any:
        """Execute a specific capability"""
        if capability == "load_data":
            return self._load_data(**kwargs)
        elif capability == "clean_data":
            return self._clean_data(**kwargs)
        elif capability == "transform_data":
            return self._transform_data(**kwargs)
        elif capability == "analyze_data":
            return self._analyze_data(**kwargs)
        elif capability == "export_data":
            return self._export_data(**kwargs)
        else:
            raise ValueError(f"Unknown capability: {capability}")
    
    def register_tools(self) -> Dict[str, callable]:
        """Register and return tool functions"""
        return {
            "load_data": self._load_data,
            "clean_data": self._clean_data,
            "transform_data": self._transform_data,
            "analyze_data": self._analyze_data,
            "export_data": self._export_data,
            "validate_data": self._validate_data,
            "merge_data": self._merge_data,
            "sample_data": self._sample_data
        }
    
    def _load_data(self, file_path: str, format: str = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load data from file"""
        start_time = time.time()
        options = options or {}
        
        try:
            file_path = Path(file_path)
            
            # Validate file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.stat().st_size > self.max_file_size:
                raise ValueError(f"File size exceeds maximum allowed size: {self.max_file_size}")
            
            # Detect format if not specified
            if not format:
                format = file_path.suffix.lower().lstrip('.')
            
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            # Load data based on format
            if format == "csv":
                df = pd.read_csv(file_path, **options)
            elif format == "json":
                df = pd.read_json(file_path, **options)
            elif format == "xlsx":
                df = pd.read_excel(file_path, **options)
            elif format == "parquet":
                df = pd.read_parquet(file_path, **options)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            load_time = time.time() - start_time
            
            return {
                "data": df.to_dict('records'),
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "load_time": load_time,
                "file_path": str(file_path),
                "format": format
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def _clean_data(self, data: Union[List[Dict], pd.DataFrame], 
                   operations: List[str] = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Clean and preprocess data"""
        start_time = time.time()
        operations = operations or ["remove_duplicates", "handle_missing"]
        options = options or {}
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            original_shape = df.shape
            cleaning_report = {}
            
            # Apply cleaning operations
            for operation in operations:
                if operation == "remove_duplicates":
                    before_count = len(df)
                    df = df.drop_duplicates()
                    after_count = len(df)
                    cleaning_report["duplicates_removed"] = before_count - after_count
                
                elif operation == "handle_missing":
                    missing_before = df.isnull().sum().sum()
                    
                    # Handle missing values based on column type
                    for column in df.columns:
                        if df[column].dtype in ['int64', 'float64']:
                            # Fill numeric columns with median
                            df[column].fillna(df[column].median(), inplace=True)
                        else:
                            # Fill categorical columns with mode
                            mode_value = df[column].mode()
                            if not mode_value.empty:
                                df[column].fillna(mode_value[0], inplace=True)
                    
                    missing_after = df.isnull().sum().sum()
                    cleaning_report["missing_values_handled"] = missing_before - missing_after
                
                elif operation == "normalize":
                    # Normalize numeric columns
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    for column in numeric_columns:
                        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                    cleaning_report["normalized_columns"] = list(numeric_columns)
                
                elif operation == "outliers":
                    # Remove outliers using IQR method
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    outliers_removed = 0
                    
                    for column in numeric_columns:
                        Q1 = df[column].quantile(0.25)
                        Q3 = df[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        before_count = len(df)
                        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                        outliers_removed += before_count - len(df)
                    
                    cleaning_report["outliers_removed"] = outliers_removed
            
            processing_time = time.time() - start_time
            rows_removed = original_shape[0] - df.shape[0]
            
            return {
                "cleaned_data": df.to_dict('records'),
                "cleaning_report": cleaning_report,
                "rows_removed": rows_removed,
                "processing_time": processing_time,
                "original_shape": original_shape,
                "final_shape": df.shape
            }
            
        except Exception as e:
            self.logger.error(f"Failed to clean data: {e}")
            raise
    
    def _transform_data(self, data: Union[List[Dict], pd.DataFrame], 
                       transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Transform and manipulate data"""
        start_time = time.time()
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            transformation_log = []
            
            # Apply transformations
            for transform in transformations:
                transform_type = transform["type"]
                parameters = transform.get("parameters", {})
                
                if transform_type == "filter":
                    # Filter rows based on conditions
                    condition = parameters.get("condition")
                    if condition:
                        # Simple condition parsing (column, operator, value)
                        column = condition.get("column")
                        operator = condition.get("operator", "==")
                        value = condition.get("value")
                        
                        if column in df.columns:
                            if operator == "==":
                                df = df[df[column] == value]
                            elif operator == "!=":
                                df = df[df[column] != value]
                            elif operator == ">":
                                df = df[df[column] > value]
                            elif operator == "<":
                                df = df[df[column] < value]
                            elif operator == ">=":
                                df = df[df[column] >= value]
                            elif operator == "<=":
                                df = df[df[column] <= value]
                            
                            transformation_log.append(f"Filtered by {column} {operator} {value}")
                
                elif transform_type == "sort":
                    # Sort data
                    columns = parameters.get("columns", [])
                    ascending = parameters.get("ascending", True)
                    
                    if columns:
                        df = df.sort_values(by=columns, ascending=ascending)
                        transformation_log.append(f"Sorted by {columns}")
                
                elif transform_type == "group":
                    # Group data
                    group_by = parameters.get("group_by", [])
                    aggregations = parameters.get("aggregations", {})
                    
                    if group_by and aggregations:
                        df = df.groupby(group_by).agg(aggregations).reset_index()
                        transformation_log.append(f"Grouped by {group_by} with {aggregations}")
                
                elif transform_type == "aggregate":
                    # Aggregate data
                    aggregations = parameters.get("aggregations", {})
                    
                    if aggregations:
                        result = {}
                        for column, agg_func in aggregations.items():
                            if column in df.columns:
                                if agg_func == "sum":
                                    result[f"{column}_sum"] = df[column].sum()
                                elif agg_func == "mean":
                                    result[f"{column}_mean"] = df[column].mean()
                                elif agg_func == "count":
                                    result[f"{column}_count"] = df[column].count()
                                elif agg_func == "min":
                                    result[f"{column}_min"] = df[column].min()
                                elif agg_func == "max":
                                    result[f"{column}_max"] = df[column].max()
                        
                        df = pd.DataFrame([result])
                        transformation_log.append(f"Aggregated with {aggregations}")
                
                elif transform_type == "pivot":
                    # Pivot data
                    index = parameters.get("index")
                    columns = parameters.get("columns")
                    values = parameters.get("values")
                    
                    if index and columns and values:
                        df = df.pivot_table(index=index, columns=columns, values=values, fill_value=0)
                        df = df.reset_index()
                        transformation_log.append(f"Pivoted with index={index}, columns={columns}, values={values}")
            
            processing_time = time.time() - start_time
            
            return {
                "transformed_data": df.to_dict('records'),
                "transformation_log": transformation_log,
                "processing_time": processing_time,
                "final_shape": df.shape
            }
            
        except Exception as e:
            self.logger.error(f"Failed to transform data: {e}")
            raise
    
    def _analyze_data(self, data: Union[List[Dict], pd.DataFrame], 
                     analysis_type: str, columns: List[str] = None) -> Dict[str, Any]:
        """Perform statistical analysis on data"""
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Select columns for analysis
            if columns:
                df = df[columns]
            
            analysis_results = {}
            statistics = {}
            insights = []
            
            if analysis_type == "descriptive":
                # Descriptive statistics
                statistics = df.describe().to_dict()
                
                # Generate insights
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for column in numeric_columns:
                    mean_val = df[column].mean()
                    median_val = df[column].median()
                    std_val = df[column].std()
                    
                    if abs(mean_val - median_val) > std_val:
                        insights.append(f"{column} shows significant skewness")
                    
                    if std_val > mean_val:
                        insights.append(f"{column} has high variability")
                
                analysis_results["descriptive_stats"] = statistics
            
            elif analysis_type == "correlation":
                # Correlation analysis
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    correlation_matrix = numeric_df.corr().to_dict()
                    analysis_results["correlation_matrix"] = correlation_matrix
                    
                    # Find strong correlations
                    for col1 in numeric_df.columns:
                        for col2 in numeric_df.columns:
                            if col1 != col2:
                                corr_val = numeric_df[col1].corr(numeric_df[col2])
                                if abs(corr_val) > 0.7:
                                    insights.append(f"Strong correlation between {col1} and {col2}: {corr_val:.2f}")
            
            elif analysis_type == "distribution":
                # Distribution analysis
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                distributions = {}
                
                for column in numeric_columns:
                    distributions[column] = {
                        "mean": df[column].mean(),
                        "median": df[column].median(),
                        "mode": df[column].mode().iloc[0] if not df[column].mode().empty else None,
                        "std": df[column].std(),
                        "skewness": df[column].skew(),
                        "kurtosis": df[column].kurtosis()
                    }
                    
                    # Distribution insights
                    skew = df[column].skew()
                    if abs(skew) > 1:
                        insights.append(f"{column} is highly skewed (skewness: {skew:.2f})")
                
                analysis_results["distributions"] = distributions
            
            elif analysis_type == "outliers":
                # Outlier detection
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                outliers = {}
                
                for column in numeric_columns:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    outliers[column] = {
                        "count": len(column_outliers),
                        "percentage": len(column_outliers) / len(df) * 100,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound
                    }
                    
                    if outliers[column]["percentage"] > 5:
                        insights.append(f"{column} has {outliers[column]['percentage']:.1f}% outliers")
                
                analysis_results["outliers"] = outliers
            
            return {
                "analysis_results": analysis_results,
                "statistics": statistics,
                "insights": insights,
                "analysis_type": analysis_type,
                "columns_analyzed": list(df.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze data: {e}")
            raise
    
    def _export_data(self, data: Union[List[Dict], pd.DataFrame], 
                    file_path: str, format: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export data to file"""
        start_time = time.time()
        options = options or {}
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format == "csv":
                df.to_csv(file_path, index=False, **options)
            elif format == "json":
                df.to_json(file_path, orient='records', **options)
            elif format == "xlsx":
                df.to_excel(file_path, index=False, **options)
            elif format == "parquet":
                df.to_parquet(file_path, **options)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            export_time = time.time() - start_time
            file_size = file_path.stat().st_size
            
            return {
                "file_path": str(file_path),
                "file_size": file_size,
                "rows_exported": len(df),
                "columns_exported": len(df.columns),
                "export_time": export_time,
                "format": format
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            raise
    
    def _validate_data(self, data: Union[List[Dict], pd.DataFrame], 
                      schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data against schema"""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "statistics": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "missing_values": df.isnull().sum().sum(),
                    "duplicate_rows": df.duplicated().sum()
                }
            }
            
            # Basic validation
            if df.empty:
                validation_results["errors"].append("Dataset is empty")
                validation_results["is_valid"] = False
            
            # Check for missing values
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 50:
                validation_results["warnings"].append(f"High percentage of missing values: {missing_percentage:.1f}%")
            
            # Check for duplicates
            duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
            if duplicate_percentage > 10:
                validation_results["warnings"].append(f"High percentage of duplicate rows: {duplicate_percentage:.1f}%")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate data: {e}")
            raise
    
    def _merge_data(self, left_data: Union[List[Dict], pd.DataFrame], 
                   right_data: Union[List[Dict], pd.DataFrame],
                   on: Union[str, List[str]], how: str = "inner") -> Dict[str, Any]:
        """Merge two datasets"""
        try:
            # Convert to DataFrames if needed
            if isinstance(left_data, list):
                left_df = pd.DataFrame(left_data)
            else:
                left_df = left_data.copy()
            
            if isinstance(right_data, list):
                right_df = pd.DataFrame(right_data)
            else:
                right_df = right_data.copy()
            
            # Perform merge
            merged_df = pd.merge(left_df, right_df, on=on, how=how)
            
            return {
                "merged_data": merged_df.to_dict('records'),
                "merge_info": {
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "merged_rows": len(merged_df),
                    "merge_key": on,
                    "merge_type": how
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to merge data: {e}")
            raise
    
    def _sample_data(self, data: Union[List[Dict], pd.DataFrame], 
                    sample_size: int = None, sample_fraction: float = None,
                    method: str = "random") -> Dict[str, Any]:
        """Sample data"""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            if sample_size:
                if sample_size > len(df):
                    sample_size = len(df)
                sampled_df = df.sample(n=sample_size, random_state=42)
            elif sample_fraction:
                sampled_df = df.sample(frac=sample_fraction, random_state=42)
            else:
                # Default to 10% sample
                sampled_df = df.sample(frac=0.1, random_state=42)
            
            return {
                "sampled_data": sampled_df.to_dict('records'),
                "sample_info": {
                    "original_rows": len(df),
                    "sampled_rows": len(sampled_df),
                    "sample_ratio": len(sampled_df) / len(df),
                    "method": method
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to sample data: {e}")
            raise


# Module metadata for discovery
MODULE_METADATA = {
    "id": "data_processor_tool",
    "name": "Data Processor Tool",
    "version": "1.2.0",
    "module_type": "tool",
    "description": "Comprehensive data processing tool with cleaning, transformation, and analysis capabilities",
    "author": "AgentKen Team",
    "license": "MIT",
    "capabilities": [
        {
            "name": "load_data",
            "description": "Load data from various file formats",
            "tags": ["data", "load", "file"]
        },
        {
            "name": "clean_data",
            "description": "Clean and preprocess data",
            "tags": ["data", "cleaning", "preprocessing"]
        },
        {
            "name": "transform_data",
            "description": "Transform and manipulate data",
            "tags": ["data", "transformation", "manipulation"]
        },
        {
            "name": "analyze_data",
            "description": "Perform statistical analysis on data",
            "tags": ["data", "analysis", "statistics"]
        },
        {
            "name": "export_data",
            "description": "Export data to various formats",
            "tags": ["data", "export", "file"]
        }
    ],
    "dependencies": [],
    "python_requirements": ["pandas>=1.3.0", "numpy>=1.21.0", "openpyxl>=3.0.0"],
    "tags": ["data", "processing", "analysis", "tool"]
}


def get_metadata():
    """Function to get module metadata"""
    return MODULE_METADATA


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Create module instance
    module = DataProcessorToolModule("data_processor_tool")
    
    # Initialize
    if module.initialize():
        print("✅ Module initialized successfully")
        
        # Test capabilities
        try:
            # Create sample data
            sample_data = [
                {"name": "Alice", "age": 25, "salary": 50000, "department": "Engineering"},
                {"name": "Bob", "age": 30, "salary": 60000, "department": "Engineering"},
                {"name": "Charlie", "age": 35, "salary": 70000, "department": "Marketing"},
                {"name": "Diana", "age": 28, "salary": 55000, "department": "Marketing"},
                {"name": "Eve", "age": 32, "salary": 65000, "department": "Engineering"}
            ]
            
            # Test data cleaning
            clean_result = module.execute("clean_data", data=sample_data, operations=["remove_duplicates"])
            print(f"Data cleaning completed: {clean_result['rows_removed']} rows removed")
            
            # Test data transformation
            transform_result = module.execute("transform_data", 
                                            data=sample_data,
                                            transformations=[
                                                {
                                                    "type": "filter",
                                                    "parameters": {
                                                        "condition": {
                                                            "column": "age",
                                                            "operator": ">",
                                                            "value": 30
                                                        }
                                                    }
                                                }
                                            ])
            print(f"Data transformation completed: {len(transform_result['transformed_data'])} rows remaining")
            
            # Test data analysis
            analysis_result = module.execute("analyze_data", 
                                           data=sample_data,
                                           analysis_type="descriptive")
            print(f"Data analysis completed: {len(analysis_result['insights'])} insights generated")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        # Shutdown
        module.shutdown()
    else:
        print("❌ Module initialization failed")