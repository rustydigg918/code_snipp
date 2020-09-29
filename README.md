# code_snipp

# Projects

## Research

### Getting into a new project

* What is the purpose of this new project? Is it clearly defined?
* What value is unlocked at v1? Can this value really be deployed? What second order effects might occur _after_ deployment (and what _blocks_ the value from being released)?
* What's the biggest uncertainty ahead? What does it cost to solve it, what value does it bring?
  * Try to derisk the biggest/most painful uncertainties soonest
* What single thing might give us 80% of the remaining answer? Probably avoid the edge cases (document, but don't focus on them)
  * Identify the most valuable deliverables (most-derisking, most-value-yielding) to rank opportunities
* Milestones - list some, note what each enables

## Strategy

* Prefer who/what/why/where/when objective questions
* Focus on understanding the levers for highest value delivery, check for shared agreement on these (ignore low-value stuff that's cool but won't enable beneficial change)
* Aim for frequent mini-retros to discuss "what did we learn? what shouldn't we repeat?"

# Definitions

* Billion in UK and USA is 10^9 (the "short billion"), unlike various parts of Europe. Milliard can be used in Europe to refer to 10^9.

# Python

## Wants

* Pandas better describe - colour the percentiles, not the counts/mean/std, include 5/95% in percentiles. Add dtype description and maybe memory, not how many columns were ignored too (stick on the end maybe?)

* Pandas better cut - give it some ranges and ask for nice labels and it'll form e.g. in scientific form (1M-500k ...) with open/closed labels, maybe special handling of e.g. 0, with formatting for currency and others

* Matplotlib label formatter - take int/float labels and convert to eg currency (2dp), human readable (e.g. 1M), optional leading symbol (e.g. £, $) or trailing text (e.g. pp.), with commas (e.g. "2,000") `friendly_label(dp=2, leading_text="", following_text="", with_commas=False, ints_if_possible=False)` and `human_readable(...)`

## TKINTER

```
.grid() 	    .pack()
sticky="ns" 	fill=tk.Y
sticky="ew" 	fill=tk.X
sticky="nsew" 	fill=tk.BOTH
```

```
You can change the location of each label inside of the grid cell using the sticky parameter. sticky accepts a string containing one or more of the following letters:

    "n" or "N" to align to the top-center part of the cell
    "e" or "E" to align to the right-center side of the cell
    "s" or "S" to align to the bottom-center part of the cell
    "w" or "W" to align to the left-center side of the cell

The letters "n", "s", "e", and "w" come from the cardinal directions north, south, east, and west.
```

### Responsive biscuits
```
import tkinter as tk

window = tk.Tk()

for i in range(3):
    window.columnconfigure(i, weight=1, minsize=75)
    window.rowconfigure(i, weight=1, minsize=50)

    for j in range(0, 3):
        frame = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame.grid(row=i, column=j, padx=5, pady=5)

        label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
        label.pack(padx=5, pady=5)

window.mainloop()
```

### Sticky: North-East(ne) and South-West(sw)

```
import tkinter as tk

window = tk.Tk()
window.columnconfigure(0, minsize=250)
window.rowconfigure([0, 1], minsize=100)

label1 = tk.Label(text="A")
label1.grid(row=0, column=0, sticky="ne")

label2 = tk.Label(text="B")
label2.grid(row=1, column=0, sticky="sw")

window.mainloop()
```

### Responsive Form
```
import tkinter as tk

# Create a new window with the title "Address Entry Form"
window = tk.Tk()
window.title("Address Entry Form")

# Create a new frame `frm_form` to contain the Label
# and Entry widgets for entering address information.
frm_form = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
# Pack the frame into the window
frm_form.pack()

# Create the Label and Entry widgets for "First Name"
lbl_first_name = tk.Label(master=frm_form, text="First Name:")
ent_first_name = tk.Entry(master=frm_form, width=50)
# Use the grid geometry manager to place the Label and
# Entry widgets in the first and second columns of the
# first row of the grid
lbl_first_name.grid(row=0, column=0, sticky="e")
ent_first_name.grid(row=0, column=1)

# Create the Label and Entry widgets for "Last Name"
lbl_last_name = tk.Label(master=frm_form, text="Last Name:")
ent_last_name = tk.Entry(master=frm_form, width=50)
# Place the widgets in the second row of the grid
lbl_last_name.grid(row=1, column=0, sticky="e")
ent_last_name.grid(row=1, column=1)

# Create the Label and Entry widgets for "Address Line 1"
lbl_address1 = tk.Label(master=frm_form, text="Address Line 1:")
ent_address1 = tk.Entry(master=frm_form, width=50)
# Place the widgets in the third row of the grid
lbl_address1.grid(row=2, column=0, sticky="e")
ent_address1.grid(row=2, column=1)

# Create the Label and Entry widgets for "Address Line 2"
lbl_address2 = tk.Label(master=frm_form, text="Address Line 2:")
ent_address2 = tk.Entry(master=frm_form, width=5)
# Place the widgets in the fourth row of the grid
lbl_address2.grid(row=3, column=0, sticky=tk.E)
ent_address2.grid(row=3, column=1)

# Create the Label and Entry widgets for "City"
lbl_city = tk.Label(master=frm_form, text="City:")
ent_city = tk.Entry(master=frm_form, width=50)
# Place the widgets in the fifth row of the grid
lbl_city.grid(row=4, column=0, sticky=tk.E)
ent_city.grid(row=4, column=1)

# Create the Label and Entry widgets for "State/Province"
lbl_state = tk.Label(master=frm_form, text="State/Province:")
ent_state = tk.Entry(master=frm_form, width=50)
# Place the widgets in the sixth row of the grid
lbl_state.grid(row=5, column=0, sticky=tk.E)
ent_state.grid(row=5, column=1)

# Create the Label and Entry widgets for "Postal Code"
lbl_postal_code = tk.Label(master=frm_form, text="Postal Code:")
ent_postal_code = tk.Entry(master=frm_form, width=50)
# Place the widgets in the seventh row of the grid
lbl_postal_code.grid(row=6, column=0, sticky=tk.E)
ent_postal_code.grid(row=6, column=1)

# Create the Label and Entry widgets for "Country"
lbl_country = tk.Label(master=frm_form, text="Country:")
ent_country = tk.Entry(master=frm_form, width=50)
# Place the widgets in the eight row of the grid
lbl_country.grid(row=7, column=0, sticky=tk.E)
ent_country.grid(row=7, column=1)

# Create a new frame `frm_buttons` to contain the
# Submit and Clear buttons. This frame fills the
# whole window in the horizontal direction and has
# 5 pixels of horizontal and vertical padding.
frm_buttons = tk.Frame()
frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)

# Create the "Submit" button and pack it to the
# right side of `frm_buttons`
btn_submit = tk.Button(master=frm_buttons, text="Submit")
btn_submit.pack(side=tk.RIGHT, padx=10, ipadx=10)

# Create the "Clear" button and pack it to the
# right side of `frm_buttons`
btn_clear = tk.Button(master=frm_buttons, text="Clear")
btn_clear.pack(side=tk.RIGHT, ipadx=10)

# Start the application
window.mainloop()
```
**or**
```
import tkinter as tk

# Create a new window with the title "Address Entry Form"
window = tk.Tk()
window.title("Address Entry Form")

# Create a new frame `frm_form` to contain the Label
# and Entry widgets for entering address information.
frm_form = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
# Pack the frame into the window
frm_form.pack()

# List of field labels
labels = [
    "First Name:",
    "Last Name:",
    "Address Line 1:",
    "Address Line 2:",
    "City:",
    "State/Province:",
    "Postal Code:",
    "Country:",
]

# Loop over the list of field labels
for idx, text in enumerate(labels):
    # Create a Label widget with the text from the labels list
    label = tk.Label(master=frm_form, text=text)
    # Create an Entry widget
    entry = tk.Entry(master=frm_form, width=50)
    # Use the grid geometry manager to place the Label and
    # Entry widgets in the row whose index is idx
    label.grid(row=idx, column=0, sticky="e")
    entry.grid(row=idx, column=1)

# Create a new frame `frm_buttons` to contain the
# Submit and Clear buttons. This frame fills the
# whole window in the horizontal direction and has
# 5 pixels of horizontal and vertical padding.
frm_buttons = tk.Frame()
frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)

# Create the "Submit" button and pack it to the
# right side of `frm_buttons`
btn_submit = tk.Button(master=frm_buttons, text="Submit")
btn_submit.pack(side=tk.RIGHT, padx=10, ipadx=10)

# Create the "Clear" button and pack it to the
# right side of `frm_buttons`
btn_clear = tk.Button(master=frm_buttons, text="Clear")
btn_clear.pack(side=tk.RIGHT, ipadx=10)

# Start the application
window.mainloop()
```
### Basic app architecture
```
import tkinter as tk

def increase():
    value = int(lbl_value["text"])
    lbl_value["text"] = f"{value + 1}"


def decrease():
    value = int(lbl_value["text"])
    lbl_value["text"] = f"{value - 1}"

window = tk.Tk()

window.rowconfigure(0, minsize=50, weight=1)
window.columnconfigure([0, 1, 2], minsize=50, weight=1)

btn_decrease = tk.Button(master=window, text="-", command=decrease)
btn_decrease.grid(row=0, column=0, sticky="nsew")

lbl_value = tk.Label(master=window, text="0")
lbl_value.grid(row=0, column=1)

btn_increase = tk.Button(master=window, text="+", command=increase)
btn_increase.grid(row=0, column=2, sticky="nsew")

window.mainloop()
```

## Pandas

### `groupby`

`gpby = df.groupby` generates a `groupby.generic.DataFrameGroupBy`. This has a `__len__`, `gpby.groups` shows the a dict of keys for the groups and the indices that match the rows.

`for group in gpby:` generates a tuple of `(name, subset_dataframe)`, the `subset_dataframe` has all the columns including the group keys. `.groups` generates this entire list of groups as a dict, index into it using e.g. `gpby.groups[('-', '-', 'Canada')]` to retrieve the indices (and use `df.loc[indices]`) to fetch the rows.  `gpby.get_group(('-',  '-',  'Canada'))` takes the keys and returns a `DataFrame` without the keys.

`gpby.transform` works on `Series` or `DataFrame` and returns a `Series` (?) that matches each input row pre-grouping. `gpby.apply` returns a result based on the grouped items.

`gpby` calls such as `mean` are delegated.

`groupby` on a timeseries (e.g. day index) won't care about missing items (e.g. missing days), using `resample` for date times with a `fillna` is probably more sensible.

Categoricals have a non-obvious memory behaviour in 1.0 in `groupby`, must pass in `observed=True` on the `groupby` else it stalls and eats a lot of RAM: https://github.com/pandas-dev/pandas/issues/30552 It builds the cartesian product of all possible categorical groups with the default arguments which might take some time and RAM.

### `crosstab`

`normalize` can take `"index"`/`0` (note not `"rows"`), `"columns"`/`1` or `"all"`/`True`, default is `False`.

### `pd.to_datetime`

`utc=True` will set timezone (else no tz info). Lowest valid date we can parse is circa `pd.to_datetime('1677-09-22', utc=True)` (21st will raise a `OutOfBoundsDatetime` unless `errors="ignore"` passed, if this is passed then we get a string back in place! use `coerce` to get `NaT` for invalid times) - limitations: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timestamp-limitations . Error handling: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#invalid-data

### `extract`
Pandas is full of text management tool to implement to extract capture groups in the regex pattern as columns in a DataFrame: (official documentation of the feature, e.g {`tmp['month'] = tmp.FileName.str.extract('(jun|july|aug|mar|apr|may)', expand = False).str.strip()`} Find its params here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html)

### `cut`

Pandas is closed-right by default i.e. with `right=True` (default) then bins are `(b1, b2]` (exclusive/open of left, inclusive/closed of right: https://en.wikipedia.org/wiki/Bracket_(mathematics) ). https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html

### `merge`

Merging is the underlying operation, `df.join` is a shortcut into `merge`.  `join` merges keys on the left with the index on the right, it doesn't change the resulting index. Prefer `merge` to stay explicit.

`indicator=True` adds `_merge` column with indicators like `both`. `validate='many_to_one'` validates uniqueness on the right (or left or both), raising `MergeError` if not validated.

### `concatenate`

Has lots of options including to drop the index, `axis=1` for columnar concatenation and more.

### `reindex`

Apply a new index e.g. a series of ints to a non-contiguous existing `index`, `fill_na` defaults to `NaN`.

### `value_counts`

Probably add `dropna=False` every time _caveat_ this means we easily miss NaN values, the sanity_check is to count the results and check them against the size of the original column.

### `info`

`df.info(memory_usage="deep")` introspects each column and counts bytes used including strings - but this can be slow on many strings (e.g. millions of rows of strings might take 1 minute).

### `describe`

* `ser.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99, 1])` add more percentiles.
* `df.describe().style.background_gradient(axis=0)`

### `read_csv`

`pd.read_csv(parse_dates=True)` will only parse index dates, instead use `parse_dates=['col1', 'col2']` to parse other cols.

### `str.contains`

Lightweight pattern match use to e.g. `df.columns.contains('somestring', ignorecase=True)` to find substring 
`somestring` in column names, returns a mask.

### concatenate list items
```
>>> sentence = ['this','is','a','sentence']
>>> '-'.join(sentence)
'this-is-a-sentence'
```
**or**
```
>>> my_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> my_lst_str = ''.join(map(str, my_lst))
>>> print(my_lst_str)
'12345678910'
```
**or**
```
from str import join

sentence = ['this','is','a','sentence']

join(sentence, "-") 
      **or**
from functools import reduce

sentence = ['this','is','a','sentence']
out_str = str(reduce(lambda x,y: x+"-"+y, sentence))
print(out_str)
```
**or**
```
arr=['a','b','h','i']     # let this be the list
s=""                      # creating a empty string
for i in arr:
   s+=i                   # to form string without using any function
print(s) 
```
**or**
```

Without .join() :

my_list=["this","is","a","sentence"]

concenated_string=""
for string in range(len(my_list)):
    if string == len(my_list)-1:
        concenated_string+=my_list[string]
    else:
        concenated_string+=f'{my_list[string]}-'
print([concenated_string])
>>> ['this-is-a-sentence']

```
### dict to dataframe
```
my_dict = {key:value,key:value,key:value,...}
df = pd.DataFrame(list(my_dict.items()),columns = ['column1','column2']) 
```
### Muliple file merging
```
import pandas as pd
import glob

path = r'C:\DRO\DCL_rawdata_files' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
```
**or**
```
path = r'C:\DRO\DCL_rawdata_files'                     # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
# doesn't create a list, nor does it append to one
```
**or**
```
import glob, os    
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "my_files*.csv"))))
```
**or**
```
For a few files - 1 liner:

df = pd.concat(map(pd.read_csv, ['data/d1.csv', 'data/d2.csv','data/d3.csv']))

For many files:

from os import listdir

filepaths = [f for f in listdir("./data") if f.endswith('.csv')]
df = pd.concat(map(pd.read_csv, filepaths))
```
**or**
```
import glob

df = pd.concat(map(pd.read_csv, glob.glob('data/*.csv')))
```

```
from glob import iglob
import pandas as pd

path = r'C:\user\your\path\**\*.csv'

all_rec = iglob(path, recursive=True)     
dataframes = (pd.read_csv(f) for f in all_rec)
big_dataframe = pd.concat(dataframes, ignore_index=True)

Note that the three last lines can be expressed in one single line:

df = pd.concat((pd.read_csv(f) for f in iglob(path, recursive=True)), ignore_index=True)
```

### Vaidating the existance of a sheet in an excel workbook
```from openpyxl import load_workbook
 
wb = load_workbook(file_workbook, read_only=True)   # open an Excel file and return a workbook
    
if 'sheet1' in wb.sheetnames:
    print('sheet1 exists')
```
**Note:** The Python library openpyxl is designed for reading and writing Excel xlsx/xlsm/xltx/xltm files. The following snippet code checks if a specific sheet name exists in a given workbook.

*For older Microsoft Excel files (i.e., .xls), use xlrd and xlwt instead.*


### To remove white spaces:
```
1) To remove white space everywhere:

df.columns = df.columns.str.replace(' ', '')

2) To remove white space at the beginning of string:

df.columns = df.columns.str.lstrip()

3) To remove white space at the end of string:

df.columns = df.columns.str.rstrip()

4) To remove white space at both ends:

df.columns = df.columns.str.strip()

    To replace white spaces with other characters (underscore for instance):

5) To replace white space everywhere

df.columns = df.columns.str.replace(' ', '_')

6) To replace white space at the beginning:

df.columns = df.columns.str.replace('^ +', '_')

7) To replace white space at the end:

df.columns = df.columns.str.replace(' +$', '_')

8) To replace white space at both ends:

df.columns = df.columns.str.replace('^ +| +$', '_')

All above applies to a specific column as well, assume you have a column named col, then just do:

df[col] = df[col].str.strip()  # or .replace as above
```

### find index of value anywhere in DataFrame
```
for row in range(df.shape[0]): # df is the DataFrame
         for col in range(df.shape[1]):
             if df.get_value(row,col) == 'security_id':
                 print(row, col)
                 break
```
### reading Excel/CSV file starting from the row below that with a specific value
```

df = pd.read_excel('your/path/filename')
```

**This answer helps in finding the location of 'start' in the df**
```
 for row in range(df.shape[0]): 

       for col in range(df.shape[1]):

           if df.iat[row,col] == 'start':

             row_start = row
             break
```

**after having row_start you can use subframe of pandas**
```
df_required = df.loc[row_start:]
```
**And if you don't need the row containing 'start', just u increment row_start by 1**
```
df_required = df.loc[row_start+1:]
```

### Converting data while loading by calling a function in `converters`
```
file = "Books.xls"
def convert_author_cell(cell):
    if cell == "Hilary":
        return 'visly'
    return cell
data = pd.read_excel(file,converters={'Author':convert_author_cell})
```
### formatting

`vc.apply(lambda v: f"{v*100:.0f}%")` turn a `value_counts` into percentages like "92%"

### display options

`pd.get_option('display.max_columns')` probably 20 and `max_rows` is 60. Use `pd.set_option('display.max_columns', 100)` for more cols in e.g. `.head()`. `pd.set_option('precision', 4)`. https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#frequently-used-options

```
def show_all(many_rows, max_rows=999):
    """Show many rows rather than the typical small default"""
    from IPython.display import display
    with pd.option_context('display.max_rows', 999):
        display(many_rows) # display required else no output generated due to indentation
```

### Data processing tips

`query` with `NaN` rows is a pain, for text columns we could replace missing data with `-` and then that's another string-like thing for a query, this significantly simplifies the queries.

## matplotlib

### my style (?)

```
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x') # horizontal grid lines only
ax.legend(frameon=False)
```

### `subplot`

* `fig, ax = plt.subplots()`
* `fig, axs = plt.subplots(figsize=(20, 12), nrows=2, gridspec_kw={'height_ratios': [2, 1]})` 

### Pandas `plot`

Marks with x and no lines: `.plot(marker='x', linestyle=' ', ax=ax)`

## Horiztonal lines

`ax.axhline(color='grey', linestyle='--')`


### subplots

* `fig, axs = plt.subplots(ncols=2, figsize=(8, 6))`

### limits

#### symmetric limits (x and y have same range)

```
min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
max_val = min(ax.get_xlim()[1], ax.get_ylim()[1])
ax.set_xlim(xmin=min_val, xmax=max_val)
ax.set_ylim(ymin=min_val, ymax=max_val)
```

#### symmetric (y axis)

```
biggest = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
ax.set_ylim(ymin=-biggest, ymax=biggest)
```

### axis labels

```
import matplotlib as mpl
def set_commas(ax, on_x_axis=True):
    """Add commas to e.g. 1,000,000 on axis labels"""
    axis = ax.get_xaxis()
    if not on_x_axis:
        axis = ax.get_yaxis()
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
```

```
# add e.g. s for seconds labeling on tick label
locs =ax.get_yticks()
new_yticks=[f"{d}s" for d in locs]
ax.set_yticklabels(new_yticks); 
```

* "percentage point, pp" for percentage differences https://en.wikipedia.org/wiki/Percentage_point, possibly use "proportion" in the title and aim for "%" symbol on numeric axis

### axis ticks

* https://matplotlib.org/3.1.0/api/ticker_api.html
  * `FixedLocator` is good for equal indices e.g. week count 10, 11, 12
  * `MaxNLocator` guesses at a good start/end point for fixed intervals
  * https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html examples

## Argument parsing

### `argparse`

```
parser = argparse.ArgumentParser(description=__doc__) # read __doc__ attribute
parser.add_argument('-i', '--input_filename', type=str, nargs="?",
                    help='csv from someone'
                    ' (default: %(default)s))', # help msg 2 over lines with default
                    default=input_filename) # some default
parser.add_argument('-v', '--version', action="store_true",
                    help="show version")

args = parser.parse_args()
print("Arguments provided:", args)
if args.version:
    print(f"Version: {__version__}")
```

When argument parsing it might make sense to check for the presence of specified files and to `sys.exit(1)` with a message if they're missing. `argparse` can also stream `stdin` in place of named files for piped input.

## Testing

### `unittest`

`python -m unittest` runs all tests with autodiscovery, `python -m unittest mymodule.MyClass` finds `mymodule.py` and runs the tests in `MyClass`.

There are some visual diffs but they're not brilliant. I don't think we can invoke `pdb` on failures without writing code?

### `pytest`

More mature than `unittest`, doesn't need the `unittest` methods for checking same/different code and catching expected exceptions is neater. Also has plugins, syntax colouring.

Typically we'd write `pytest` to execute it, there's something weird with being unable to find imported modules if `__init__.py` is (or maybe isn't) present, in which case `python -m pytest` does the job: https://stackoverflow.com/questions/41748464/pytest-cannot-import-module-while-python-can

`pytest --pdb` drops into the debugger on a failure. 

### `coverage`

`$ coverage run -m unittest test_all.py` (or e.g. `discover` to discover all test files) writes an sqlite3 `.coverage` datafile, `$ coverage report html` generates `./htmlcov/` and `firefox htmlcov/index.html` opens the html report. Notes: https://coverage.readthedocs.io/en/coverage-5.1/

## Profiling

### `ipython_memory_usage`

`import ipython_memory_usage; #%ipython_memory_usage_start` for cell by cell memory analysis

## Debugging

### `pdb`

In IPython `pdb.run("scipy.stats.power_divergence(o, axis=None, ddof=2, lambda_='log-likelihood')")` will invoke `pdb`, use `s` to step into the function, `n` for the next line, `p` to print state (use this to see that `f_exp` is calculated in an unexpected way, see Statistical tests below). `b _count` will set a breakpoint for the `_count` function inside `power_divergence`, run to it with `c`.


## Statistical tests

`scipy.stats.chi2_contingency(o)` on a 2D array calculates `e=scipy.stats.contingency.expected_freq(o)` internally. With a `2x2` table the `dof==1` so `correction` is used which adjusts `o`, then this calls through to `scipy.stats.power_divergence`. To use a G Test pass `lambda_="log-likelihood"` to `chi2_contingency`. To avoid the correction when debugging confirm that `scipy.stats.chi2_contingency(o, correction=False)` and `scipy.stats.power_divergence(o, e, axis=None, ddof=2)` are the same. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html .  To calculate an equivalent G Test manually use `scipy.stats.power_divergence(o, e, axis=None, ddof=2, lambda_='log-likelihood')`. _Note_ that `e` is calculated implicitly as the mean of the array inside `power_divergence` which is _not_ the same as calling `expected_freq`! Prefer to be explicit. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.power_divergence.html


## Getting to high code quality

### `flake8`

Lints and checks code, works in most IDEs and as a git commit hook. Can disable a line with annoying warnings using `  # NOQA` as the comment.

### `pylama`

Has more than `flake8`, not as widely supported (2020-05 not in Visual Code as a default linter).

### `black`

Highly opinionated, not necessarily "what you want" as some of the reformating is hard to read _but_ you also don't get a choice and that's a really good outcome!

### `bulwark`

Check dataframe cols as I go

### `watermark`

* https://github.com/rasbt/watermark

```
%load_ext watermark
%watermark -i -v -m -p pandas,numpy,matplotlib -g -b
# possible -iv for all imported pkg versions? try this...
```

# Shell

## link

`ln -s other_folder_file .` link the other file into this folder.

# File Management

## Showing all the files with its directory extension inside a master directory

  ```
    for path, subdirs, files in os.walk(r"C:\Users\pushkar\Downloads\Puma_Audit Master - Copy"):
      for name in files:
          file_path_main = os.path.join(path, name)
          print(file_path_main)
  ```



# Jupyter Notebook

## Notebook Formatting
```
  pd.set_option('display.max_rows', None)

  pd.set_option('display.max_columns', None)

  pd.set_option('display.width', None)

  pd.set_option('display.max_colwidth', None)
```

# EDA

### Univeriate Analysis
```
<Bar Plot>
fig, axes = plt.subplots(5, 2, figsize=(14, 22))
axes = [ax for axes_row in axes for ax in axes_row]

for i, c in enumerate(train[cat_cols]):
    _ = train[c].value_counts()[::-1].plot(kind = 'pie', ax=axes[i], title=c, autopct='%.0f', fontsize=12)
    _ = axes[i].set_ylabel('')
    
_ = plt.tight_layout()
```

```
<Count Plot>
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = [ax for axes_row in axes for ax in axes_row]

for i, c in enumerate(train[cat_cols]):
    _ = train[c].value_counts()[::-1].plot(kind = 'barh', ax=axes[i], title=c, fontsize=14)
    
_ = plt.tight_layout()
```


