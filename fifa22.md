# Exploración de Datos FIFA22

# 1. Introducción

Este análisis tiene como objetivo explorar el conjunto de datos de FIFA22, 
centrándose en variables clave como el potencial, salario, reputación, valor de mercado y estatura de los jugadores.


# 2. Carga y exploración inicial de datos


```python
# 2.1 Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
```

```python
# 2.2 Carga del dataset

current_folder = os.getcwd()

pickle_dir = os.path.join(current_folder, "data")
if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)
pickle_file = os.path.join(pickle_dir, "players_22.pkl")

if os.path.exists(pickle_file):
    print(f"The pickle file already exists at: {pickle_file}")
else:
    os.environ["KAGGLEHUB_CACHE"] = current_folder + "/kagglehub_cache"
    path = kagglehub.dataset_download(
        "stefanoleone992/fifa-22-complete-player-dataset"
    )
    print("Downloaded path: ", path)
    output_file = os.path.join(path, "players_22.csv")
    pd.read_csv(output_file).to_pickle(pickle_file)

df = pd.read_pickle(pickle_file)
```

```python
# 2.3 Exploración inicial
df.head()
```

```python
df.info()
```

```python
df.describe()
```

```python
# 2.4 Identificación de valores nulos
df.isnull().sum()
```

# 3. Análisis exploratorio (EDA)

## 3.1 Distribución de características físicas y económicas

```python
# Estatura de los jugadores
sns.histplot(df["height_cm"], kde=True)
plt.title("Distribución de Estatura")
plt.show()
```

```python
# Peso de los jugadores
sns.histplot(df["weight_kg"], kde=True)
plt.title("Distribución de Peso")
plt.show()
```

```python
# Salario de los jugadores
sns.histplot(df["wage_eur"], kde=True)
plt.title("Distribución de Salario")
plt.show()
```

```python
# Edad de los jugadores
sns.histplot(df["age"], kde=True)
plt.title("Distribución de Edad")
plt.show()
```

## 3.2 Dominancia del pie

```python
sns.countplot(x="preferred_foot", data=df)
plt.title("Pie Dominante de los Jugadores")
plt.show()
```

## 3.3 Representación geográfica

```python
top_countries = df["nationality_name"].value_counts().head(10)
top_countries.plot(kind="bar")
plt.title("Países con Mayor Representación")
plt.show()
```

## 3.4 Posiciones de los jugadores

```python
sns.countplot(
    y="club_position", data=df, order=df["club_position"].value_counts().index
)

plt.title("Posiciones más Comunes")

plt.show()
```

# 4. Relaciones entre variables clave

## 4.1 Potencial vs Salario

```python
sns.scatterplot(x="potential", y="wage_eur", data=df)
plt.title("Relación entre Potencial y Salario")
plt.show()
```

## 4.2 Reputación vs Valor de mercado

```python
sns.scatterplot(x="international_reputation", y="value_eur", data=df)
plt.title("Relación entre Reputación y Valor de Mercado")
plt.show()
```

## 4.3 Reputación vs Salario

```python
sns.boxplot(x="international_reputation", y="wage_eur", data=df, whis=[1,99])
plt.title("Relación entre Reputación y Salario")
plt.show()
```

## 4.4 Estatura vs Potencial

```python
sns.scatterplot(x="height_cm", y="potential", data=df)
plt.title("Relación entre Estatura y Potencial")
plt.show()
```

# 5. Jugadores destacados

## 5.1 Jugadores mejor remunerados

```python
df[["short_name", "wage_eur"]].sort_values(by="wage_eur", ascending=False).head(10)
```

## 5.2 Jugadores con mayor potencial

```python
df[["short_name", "potential"]].sort_values(by="potential", ascending=False).head(10)
```

## 5.3 Jugadores con mejor valoración general

```python
df[["short_name", "overall"]].sort_values(by="overall", ascending=False).head(10)
```

# 6. Conclusiones

- Se identificaron patrones interesantes en las relaciones entre potencial, salario y reputación.
- La mayoría de los jugadores tienden a ser diestros.
- Determinados países dominan la representación de jugadores.
- Hay correlaciones positivas moderadas entre reputación y valor de mercado.
- Se pueden proponer futuros estudios como predicción del valor de mercado a partir de atributos físicos y técnicos.
