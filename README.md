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

# load the data
df = pep.load_data(‘patient_experience.xlsx’)

# lemmatize a column
bestlemma = pep.sbl(df, df['Best Part'])

# create word frequency for column
bestcounts = pep.word_list(bestlemma)

#








```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Dan](https://github.com/humdansfw/Repo1/blob/main/license)
