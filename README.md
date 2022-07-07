# Patient Experience Pipeline

PatientExperiencePipeline is a Python library for dealing with patient experience survey texts.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install patientexperiencepipeline.

```bash
pip install patientexperiencepiepline
```

## Usage

```python
import patientexperiencepipeline as pep

load_data(filename):
    ''' '''
    Create function to: 
        read data into a pandas dataframe
        dataframe is named df
    Parameters:
        filename: string
    Returns:
        df: dataframe
    '''
pep.load_data(‘patient experience.xlsx’)

# returns data
pep.clean_data(list_of_texts)

# returns 
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Dan](https://github.com/humdansfw/Repo1/blob/main/license)
