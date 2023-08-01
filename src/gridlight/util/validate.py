from pydantic import BaseModel, model_validator
from typing import List, Optional
from datetime import datetime

class GridTargetParams(BaseModel):
    area_of_interest_data: str
    population_data: str
    start_date: str
    end_date: str
    result_subfolder: str
    dev_mode: Optional[bool] = False
    tiles: str

    @model_validator(mode='after')
    def convert_date(self) -> 'GridTargetParams':
        self.start_date = datetime.strptime(self.start_date, "%Y-%m")
        self.end_date = datetime.strptime(self.end_date, "%Y-%m")

        return self


class GridLightParams(BaseModel):
    area_of_interest_data: str
    roads_data: str
    grid_truth_data: str
    power_data: Optional[str] = None
    targets_in: str
    targets_clean_in:str
    result_subfolder: str
    dev_mode: Optional[bool] = False