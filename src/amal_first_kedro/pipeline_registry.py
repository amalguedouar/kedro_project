"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from amal_first_kedro.pipelines import data_engineering as de
from amal_first_kedro.pipelines import data_science as ds



def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_engineering_pipeline =  de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "dp": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }

