

[![DOI](https://zenodo.org/badge/902649899.svg)](https://doi.org/10.5281/zenodo.14758697)


# EZFIT A Dead simple interface for fitting in python

This package is built for use by people who are new not just to python but to coding,
fitting, and programatically interacting with data. If you have experience with
EXCEL, but need to fit data using least squares fitting, this is the tool for you.

## Installation
There are four prerequisite functions for installing ezfit, pandas, numpy, matplotlib,
and scipy. The package can be installed though the terminal with the following
command.
```
pip install ezfit numpy pandas matplotlib
```

## How to Use
Import the ezfit library. This will allow you to have a simple interface for fitting
a pandas `DataFrame` to some model.

```python
import ezfit
```
### Loading Data

To start, load your data into a pandas DataFrame. Try to allways save your data as a
.csv file with one line of headers.
[Read the documentation on this](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

#### Comma Deliminated
```
x, y, yerr
0, 1, .5
1, .5, .2
...
```
You can load this data easily with the following easy command

```python
# start by importing the pandas module
import pandas as pd

# Everythin in python uses the dot notation to access attributes and functions
# we need the read_csv() function from pandas, so we will call
df = read_csv("path_to_file")    # note that you might need a full path

# lets check that the first 2 rows look correct by getting the `head` of the df
print(df.head(2))   # the print() statement is how you print something in python
```
The output should look something like this
```
   x    y  yerr
0  0  1.0   0.5
1  1  0.5   0.2
```

We can also plot the data quickly to make sure it looks right, and determine
if there is any cleaning that needs to be done.

```python
# Lets start by getting the standard python plotting library
import matplotlib.pyplot as plt

# the df.plot() function will plot the data in the dataframe in one easy go
df.plot(x = "x", y = "y", yerr="yerr")  # you can pass other parameters in too

# this will plot the collumn labeles "y" vs "x", with error bars of size "yerr"
# you can pretty this plot up if you like, but it is fine for just checking the data

plt.show()      # This will render the currently active plot
```
You might want to place this plot on a log scale, and this can be done in many
ways. For a complete list of the parameters available to you, please read up
on the [pandas plot method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)

#### Tab Deliminated & Line Skips

Now if the dataset is not collumn seperated, as is the data from CXRO, you will need to
tell pandas what seperates the collumns. Lets look at some CXRO index of refraction
data
```
 Si3N4 Density=3.44
 Energy(eV), Delta, Beta
  30.  0.274695814  0.210541397
  31.7943592  0.252507478  0.17769818
  33.6960373  0.229885429  0.150933087
```
The first row is density informaton about the material, followed by the rows for
Energy(eV), Delta, and Beta. So first we need to skip the first row of data points.
Using the `pd.read_csv()` function, we can pass in the parameter `skiprows = n` where
`n` is the number of rows we need skipped.

Now to get the data, we need to pass in a parameter telling pandas what to look for
between collumns. Using the parameter `sep = \s+` we can tell the function that there
is an unknown number of space characters between collumns. Putting this together
we have

```python
df = pd.read_csv("path_to_file", sep=r"\s+", skiprows=1)

# printing the head gives us
print(df.head(2))
```
```
Energy(eV),    Delta,          Beta
0       30.000000  0.274696  2.105414e-01
1       31.794359  0.252507  1.776982e-01
```
Now there is one issue, the collumns have trailing commas. You can solve that easily
in many ways.

```python
df.columns = ["Energy(eV)","Delta","Beta"]
# or
df.columns = [col.replace(",", "") for col in df.columns]
# or
df.rename(columns={"Energy(eV),": "Energy(eV)", "Delta,": "Delta"})
# ... you get the idea
```

Using the same methods as above, you can plot the data, and do any cleaning to remove
bad data points.

### Defining a model
Now you will need to express your mathematical model as a python function. This is the
hardest part of fitting. The syntax is rather simple, and you never need to use types
because python is a neat language. Say we have a line
$$
    f(x) = mx+b,
$$
this function maps $x\to f(x)$ using the parameters $m$ and $b$. The goal of fitting
is to find these parameters that best describe our data. Because of this we need a
python function where you can input not just the domain $x$ but also $m$ and $b$.

```python
# Function in python are created by typeng 'def' before the name of the function

def f(x, m, b):     # For the code to work, x (or your domain) must be first
    """
    Tripple quotes can be used to create a `doc string` a fancy type of comment
    that gets attatched to the top of the function. It is allways a good idea
    to comment your functions to say what they do, why they do it, and how
    to use them. For example,

    x: Domain input

    m: slope

    b: y-intercept

    returns
    y = mx + b
    """
    y = m * x + b   # Use * for multiplication and ** for exonentiation
    return y        # return is the key word to say what the function returns
```

### Fitting
Once you define a model, and load your dataset, you need to fit your data. This
can be done very easily. So I will run you though the whole process

```python
import pandas as pd
import matplotlib.pyplot as plt
import ezplot

# ══════════/ Load the Data/ ═════════════════
df = pd.read_csv("path_to_csv")

print(df.head(10))
df.plot(x = "x", y="y", yerr = "yerr")

# ══════════/ Clean the Data/ ═════════════════
# oh no the data x < 1 is bad
mask = (df["x"] > 1)
df = df[mask]

# ══════════/ Define a Model/ ═════════════════

def line(x, m, b):
    """Line function."""
    return m * x + b

# ══════════/ Fit the Data/ ═══════════════════

model, ax = df.fit(line, "x", "y", "y_err")
# this function will generate a quick plot of the fit results
plt.show()

# The model has parameters, errors, and goodness of fit
print(model)
```
```
line:
𝜒2: 88.71565403843992
reduced 𝜒2: 0.9052617759024482
m : (value=1.0858435676047251 ± 0.0497, bounds=(-inf, inf))
b : (value=-0.4650788531268627 ± 0.0903, bounds=(-inf, inf))
```

Now say you wanted to redo the fit but adding bounds and a starting value for the
slobe of the line

```python
model, ax = df.fit(line, "x", "y", "y_err", m={ "value" : 1, "min" : 0 })
# you can pass in a dictionary for each parameter in your model
print(model)
```
Now we get slightly different results
```
line:
𝜒2: 98.71565403843992
reduced 𝜒2: 1.0052617759024482
m : (value=1.158435676047251 ± 0.0497, bounds=(-inf, inf))
b : (value=-0.4650788531268627 ± 0.0903, bounds=(-inf, inf))
```
