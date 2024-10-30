This project built an ETL pipeline which helps with EHR data statistics by extracting, transforming, and loading data from electronic health records (EHRs) into a standardized format. 

### Python and dependencies
In this project, we will use Python 3.6 + environment. Please see environment.yml which contains a list of libraries needed to set environment for this project.

### Tasks
1. Descriptive Statistics
event_statistics.py computes various metrics on the data.
- Event count: Number of events recorded for a given patient. Note that every line in
the input file is an event.
- Encounter count: Count of unique dates on which a given patient visited the hospital. All the events - DIAG, LAB and DRUG - should be considered as hospital visiting events.
- Record length: Duration (in number of days) between the first event and last event
for a given patient.

2. ETL pipeline and Feature construction
etl.py extracts, transforms and saves the data.

Step - a. Compute the index date
Step - b. Filter events
Step - c. Aggregate events
- sum values for diagnostics and medication events (i.e. event id starting with DIAG
and DRUG).
- count occurences for lab events (i.e. event id starting with LAB).
Then normalize different features into the same scale using an approach like min-max normalization.
Step - d. Save in SVMLight format
If the dimensionality of a feature vector is large but the feature vector is sparse (i.e. it has only a few nonzero elements), sparse representation should be employed.

3. Predictive Modeling
A simple model is implemented in my_model.py. 
