# Pyplot Simple Usage (DF + Classes)

Minimal examples for plotting pandas DataFrames with `matplotlib.pyplot`.

## 1) Setup

```python
import pandas as pd
import matplotlib.pyplot as plt
```

## 2) Load Data

```python
df = pd.read_csv('file.csv')
```

## 3) Simplest plots from DataFrame

### Line plot
```python
df.plot(x='x_col', y='y_col', kind='line')
plt.show()
```

### Bar plot
```python
df.plot(x='category_col', y='value_col', kind='bar')
plt.show()
```

### Histogram
```python
df['value_col'].plot(kind='hist', bins=20)
plt.show()
```

### Scatter (2 numeric columns)
```python
df.plot(x='x_col', y='y_col', kind='scatter')
plt.show()
```

## 4) Color points by class (simple)

Use this when you have a class column like `target`.

```python
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}

for cls in df['target'].unique():
    part = df[df['target'] == cls]
    plt.scatter(part['x_col'], part['y_col'],
                c=colors.get(cls, 'gray'), label=cls)

plt.xlabel('x_col')
plt.ylabel('y_col')
plt.legend(title='target')
plt.show()
```

## 5) Even shorter class coloring (numeric class labels)

If class labels are already numeric (`0,1,2...`):

```python
plt.scatter(df['x_col'], df['y_col'], c=df['target'], cmap='viridis')
plt.colorbar(label='target')
plt.show()
```

## 6) Tiny exam checklist

- Pick `x` and `y` columns.
- If classes exist: color by class and add legend.
- Always finish with `plt.show()`.
