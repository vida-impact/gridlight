from pydantic import BaseModel, model_validator
from typing import List, Optional
from datetime import datetime

class GridTargetParams(BaseModel):
    area_of_interest_data: str
    population_data: str
    nightlight_data: str
    start_date: str
    end_date: str
    result_subfolder: str
    dev_mode: Optional[bool] = True

    @model_validator
    def validate_dates(self) -> 'GridTargetParams':
        self. start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        if self.start_date > self.end_date:
            raise ValueError('end date cannot be earlier than start date')
        
        return self


