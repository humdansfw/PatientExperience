# Patient Experience Pipeline

The PatientExperiencePipeline is a Python library for dealing with patient experience survey texts.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install..

```bash
save patientexperiencepipeline.py to your current working directory
```

## Usage

The patient experience data being injested should have four columns. They are:
- "Date" (datetimes) - YYYY-MM-DD HH:MM:SS.MS
- "Best Part" (object) - string/text
- "Worst Part" (object) - string/text
- "Suggestions" (object) - string/text

Not all functions are shown as many are wrapped in larger functions in order ot automate processes. The patientexperienceanalysis3.ipynb file demonstrates usage.

```python
import patientexperiencepipeline as pep

# load the data into a pandas dataframe
df = pep.load_data(‘patient_experience.xlsx’)

# clean and lemmatize a column
bestlemma = pep.sbl(df, df['Best Part'])
# create word frequency for column
bestcounts = pep.word_list(bestlemma)
bestcounts.head()

# create a frequency distribution and word cloud for each quarter of the year.
pep.quarterclouds(df)

# create intertopic distance map
pep.LDApipe(df)

# clean dataframe and add vader scores
cdf = pep.clean_df(df)
pep.vader_df(cdf)
cdf.head()

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Dan](https://github.com/humdansfw/Repo1/blob/main/license)
