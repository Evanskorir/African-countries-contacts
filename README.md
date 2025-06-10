# Age-Structured Social Contact Patterns and Epidemic Risk in Africa

## Introduction
This repository analyzes contact matrices from 53 African countries using Adjusted Population Value Decomposition (APVD) 
combined with hierarchical clustering to uncover regional similarities in social mixing behavior. These cluster-derived 
contact patterns are integrated with an age-structured epidemiological model to simulate standardized COVID-19 epidemic 
scenarios across countries. 

## Data Files
```
data/
  ├── age_distribution # Population data for each age group used in the model.
  ├── contact_matrices # contact home, school, work, other files   # Contact matrices for various settings: Home, School,
   Work, and Other.
  ├── countries.geojson # json file containing coordinates i.e latitudes and longitudes.
  ├── african_countries.json # json file containing a list of African countries.
  ├── western_sahara.json # json file containing western sahara country coordinates i.e latitudes and longitudes.
  └── model_parameters # JSON files containing model parameters.
```

## Folder Structure
```
data                
src                    
 ├── model       
 │   ├── epid_model
 │   ├── model_base 
 │   ├── r0_generator
 │   ├── r0_generator_base
 │   └── simulation
 ├── apvd 
 ├── clustering 
 ├── contact_mtx_manipulator 
 ├── coordinatesloader
 ├── dataloader
 ├── pipeline 
 └── plotter
main 
README
```

## File Details
#### `src/model/`
- **`epid_model.py`**: Provides the model equations of the age-structured model
- **`model_base.py`**: Base class for implementing the model equations
- **`r0_generator_base.py`**: Provides a foundational framework for R0 calculations in epidemic models.
- **`r0.py`**: Calculates R0 values using the Next Generation Matrix (NGM).
- **`simulation.py`**: Simulates the model using transmission rate by considering population proportion of the 
infected, icu, hospitalized, or deaths.
#### `src/`
- **`apvd.py`**: Applies dimension reduction to the 16 by 16 scaled contact matrices to get a lower dimension, i.e 2 by 2 
for each country.
- **`clustering.py`**: Applies hierarchical clustering using the reduced dimensions to group the countries.
- **`contact_mtx_manipulator.py`**: Scales the contact matrices using the country's transmission rates and 
get lower dimension scaled contact matrices.
- **`coordinatesloader.py`**: Loads the json file that contains the latitudes and longitudes for the countries.
- **`dataloader.py`**:  Loads the contact matrices, age data, and the model parameters.
- **`pipeline.py`**: Handles data loading and initialization, setting up parameters and age groups for simulations.
- **`plotter.py`**: Generates the necessary plots for the study including contact matrices, simulation and clustering.
- **`main.py`**: Imports and initializes **`pipeline.py`** to load simulation data, and run the simulations.

## Implementation

To run the framework, follow these steps:
1. Open `main.py` and configure the simulation parameters in the **`pipeline.py`**. 
2. Run the simulation.

## Output
```
figures/countries
     ├── Selected countries e.g Benin, DR Congo, etc
     │    └── CM.pdf # Visualizes the symmetric contact input as matrix using the selected scaling.    
     ├── contact_matrices
     │    └── contact_matrices.pdf # Visualizes both scaled and unscaled contact matrices at Home, School, Work, Other,
      and Full for the selected countries (Benin, DR_Congo, Namibia, Tanzania) .
     ├── african_map.pdf  # Map of Africa colored based on the clusters, with simulated data.
     ├── deaths_vs_beta_scatter.pdf  # Scatter plot for the simulated deaths against the transmission rates.
     ├── dendogram_with_threshold.pdf  # Dendogram that displays the clusters
     ├── distance_matrix_ordered.pdf  # Ordered distance matrix using complete distance from the clustring algorithm.
     └── distance_matrix_raw.pdf  # Distance matrix calculated using euclidean distance based on the scaled reduced contact matrices.
```

## Requirement
This project is developed and tested with Python 3.8 or higher. Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
