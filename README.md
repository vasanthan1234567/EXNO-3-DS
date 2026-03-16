## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```python
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('data.csv')
df
```

<img width="382" height="270" alt="image" src="https://github.com/user-attachments/assets/e6b162c4-6f42-41f8-9e9a-a680551008fd" />


```python
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```

<img width="201" height="167" alt="image" src="https://github.com/user-attachments/assets/5842e538-1477-4ec3-872a-0a0422dadbc2" />


```python
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df
```

<img width="430" height="258" alt="image" src="https://github.com/user-attachments/assets/2369811f-ce86-4391-8fe3-240081ddaab6" />


```python
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```

<img width="397" height="267" alt="image" src="https://github.com/user-attachments/assets/7ab16cd3-9532-49ca-9420-530725499258" />


```python
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```

<img width="376" height="252" alt="image" src="https://github.com/user-attachments/assets/a857019d-1a77-4ae0-adb4-53e8a798171d" />

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
```

<img width="682" height="256" alt="image" src="https://github.com/user-attachments/assets/83d01326-ff11-4edd-bc63-2a1165a9f6fe" />


```python
pd.get_dummies(df,columns=['City'])
```

<img width="658" height="291" alt="image" src="https://github.com/user-attachments/assets/6175fc19-9935-40c9-b1dd-86850f0b2eb7" />


```python
from category_encoders import BinaryEncoder
df = pd.read_csv('data.csv')
df
````

<img width="362" height="261" alt="image" src="https://github.com/user-attachments/assets/895c1abe-b74b-4742-9c7a-947bc6e28f34" />


```python
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
df
```

<img width="367" height="257" alt="image" src="https://github.com/user-attachments/assets/1bc31ade-1dbe-4e7b-aac3-4f9c5070f6d2" />


```python
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
```

<img width="418" height="257" alt="image" src="https://github.com/user-attachments/assets/6894a913-cdd9-46e3-8a17-25bb89dffe9d" />


```python
if 'City' in CC.columns:
    CC = CC.drop('City', axis=1)
new = te.fit_transform(X = df["City"],y=df["Target"])
CC = pd.concat([CC.reset_index(drop=True),new.reset_index(drop=True)],axis = 1)
CC
```

<img width="352" height="265" alt="image" src="https://github.com/user-attachments/assets/8b141315-d5d6-4dbd-ae4b-24cf8dc943d5" />


```python
df = pd.read_csv('Data_to_Transform.csv')
df
```

<img width="581" height="310" alt="image" src="https://github.com/user-attachments/assets/fdfd64f1-ec16-4b43-ba88-20293ab157d8" />


```python
df.skew()
```


<img width="247" height="158" alt="image" src="https://github.com/user-attachments/assets/a6670e8f-f743-4530-8ec0-0f40f20cec83" />


```python
np.log(df["Highly Positive Skew"])
```

<img width="442" height="192" alt="image" src="https://github.com/user-attachments/assets/bddd3395-d66c-4be1-b2ca-0249cfe938ac" />


```python
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="442" height="201" alt="image" src="https://github.com/user-attachments/assets/dca2a837-0423-4c0f-aad3-dc3687dad4f6" />


```python
np.sqrt(df["Highly Positive Skew"])
```

<img width="433" height="207" alt="image" src="https://github.com/user-attachments/assets/3b4a04d3-2e82-4078-beb8-f294623a9911" />


```python
np.square(df["Highly Positive Skew"])
```

<img width="431" height="201" alt="image" src="https://github.com/user-attachments/assets/9ef2a921-094a-46eb-bf73-64690196ccd8" />


```python
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="737" height="338" alt="image" src="https://github.com/user-attachments/assets/e5565716-2960-4582-b601-e45f57a4a990" />


```python
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```

<img width="753" height="361" alt="image" src="https://github.com/user-attachments/assets/dc7b7359-4049-4c4f-9d23-28c30c8379bb" />


```python
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="745" height="343" alt="image" src="https://github.com/user-attachments/assets/a404a2ec-996e-4493-99f1-51ef8e8f4a70" />


```python
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
```

<img width="579" height="432" alt="download" src="https://github.com/user-attachments/assets/de87456b-5a3c-4c31-bf2b-0522b5809a6b" />

```python
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/420705a5-6f3c-4bd8-ac35-fd6716d38caa" />

```python
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/d3b0facb-64a4-4ede-af48-1eb37f412654" />


```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
```

<img width="601" height="432" alt="download" src="https://github.com/user-attachments/assets/324358dc-8ebe-484d-a9f6-47413a345238" />

```python
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/2eadc7fa-e1c0-4007-b1c2-03726d05a36f" />

```python
sm.qqplot(np.abs(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/b56bda7b-50eb-488a-8dec-61e16e6e5acc" />

```python
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="565" height="434" alt="download" src="https://github.com/user-attachments/assets/d40d0617-f287-4149-9afd-5376ef030ec7" />


```python
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/d3e948f0-6364-4767-ae48-ecae548694ed" />

```python
pd.concat([CC,new],axis = 1)
```

<img width="418" height="266" alt="image" src="https://github.com/user-attachments/assets/b5535ca2-d372-4b05-9741-b8adb4c6049c" />


# RESULT:

Thus, we have successfully performed Feature Encoding and Transformation process and saved the data to a file.

       
